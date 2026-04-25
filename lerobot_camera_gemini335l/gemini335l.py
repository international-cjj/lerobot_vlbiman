import importlib
import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.utils import get_cv2_rotation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_gemini335l import Gemini335LCameraConfig

logger = logging.getLogger(__name__)

_ALIGN_MODE_NAMES = {
    "hw": "HW_MODE",
    "sw": "SW_MODE",
}
_FRAME_AGGREGATE_OUTPUT_MODE_NAMES = {
    "full_frame_require": "FULL_FRAME_REQUIRE",
    "color_frame_require": "COLOR_FRAME_REQUIRE",
    "any_situation": "ANY_SITUATION",
}
_DEPTH_INT_PROPERTY_NAMES = {
    # Keep both new and legacy enum names for compatibility across Orbbec SDK variants.
    "disp_search_range_mode": (
        "OB_PROP_DISPARITY_SEARCH_RANGE_INT",
        "OB_PROP_DISP_SEARCH_RANGE_MODE_INT",
    ),
    "disp_search_offset": (
        "OB_PROP_DISPARITY_SEARCH_OFFSET_INT",
        "OB_PROP_DISP_SEARCH_OFFSET_INT",
    ),
}


def _import_pyorbbecsdk() -> Any:
    try:
        return importlib.import_module("pyorbbecsdk")
    except Exception:
        try:
            # Newer wheels on PyPI expose the module under pyorbbecsdk2.
            return importlib.import_module("pyorbbecsdk2")
        except Exception as exc:
            raise ImportError(
                "pyorbbecsdk is required to use Gemini335LCamera. "
                "Install the official Orbbec Python SDK first. "
                "Tried imports: pyorbbecsdk, pyorbbecsdk2."
            ) from exc


def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


class Gemini335LCamera(Camera):
    config_class = Gemini335LCameraConfig

    def __init__(self, config: Gemini335LCameraConfig):
        super().__init__(config)

        self.config = config
        self.color_mode = config.color_mode
        self.rotation = get_cv2_rotation(config.rotation)
        self.warmup_s = config.warmup_s

        self.device: Any | None = None
        self.device_info: Any | None = None
        self.pipeline: Any | None = None
        self.pipeline_config: Any | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()

        self.capture_width: int | None = config.width
        self.capture_height: int | None = config.height
        if self.width is not None and self.height is not None and self.rotation in (
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ):
            self.capture_width, self.capture_height = self.height, self.width

        self.selected_color_stream_profile: dict[str, Any] | None = None
        self.selected_depth_stream_profile: dict[str, Any] | None = None

    def __str__(self) -> str:
        identifier = self.config.serial_number_or_name or "auto"
        return f"{self.__class__.__name__}({identifier})"

    @property
    def is_connected(self) -> bool:
        return self.pipeline is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return Gemini335LCamera.list_stream_profiles()

    @staticmethod
    def list_stream_profiles(serial_number_or_name: str | None = None) -> list[dict[str, Any]]:
        ob = _import_pyorbbecsdk()
        context = ob.Context()
        device_list = context.query_devices()
        cameras: list[dict[str, Any]] = []
        selected_identifier = serial_number_or_name

        for index in range(device_list.get_count()):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()
            serial_number = info.get_serial_number()
            name = info.get_name()
            uid = info.get_uid()
            if selected_identifier is not None and selected_identifier not in {serial_number, name, uid}:
                continue

            camera_info: dict[str, Any] = {
                "name": name,
                "type": "Orbbec",
                "id": serial_number,
                "serial_number": serial_number,
                "uid": uid,
                "connection_type": info.get_connection_type(),
                "firmware_version": info.get_firmware_version(),
            }

            color_default, color_profiles, color_error = Gemini335LCamera._get_device_sensor_profiles(
                device, ob.OBSensorType.COLOR_SENSOR
            )
            if color_default is not None:
                camera_info["default_color_stream_profile"] = Gemini335LCamera._video_profile_to_dict(color_default)
                camera_info["color_stream_profiles"] = Gemini335LCamera._format_video_profiles(color_profiles)
                camera_info["color_resolution_fps"] = Gemini335LCamera._group_video_profiles(color_profiles)
            elif color_error is not None:
                camera_info["default_color_stream_profile_error"] = color_error

            depth_default, depth_profiles, depth_error = Gemini335LCamera._get_device_sensor_profiles(
                device, ob.OBSensorType.DEPTH_SENSOR
            )
            if depth_default is not None:
                camera_info["default_depth_stream_profile"] = Gemini335LCamera._video_profile_to_dict(depth_default)
                camera_info["depth_stream_profiles"] = Gemini335LCamera._format_video_profiles(depth_profiles)
                camera_info["depth_resolution_fps"] = Gemini335LCamera._group_video_profiles(depth_profiles)
            elif depth_error is not None:
                camera_info["default_depth_stream_profile_error"] = depth_error

            depth_work_modes, current_depth_work_mode = Gemini335LCamera._list_depth_work_modes_for_device(device)
            if depth_work_modes:
                camera_info["depth_work_modes"] = depth_work_modes
                camera_info["current_depth_work_mode"] = current_depth_work_mode

            cameras.append(camera_info)

        if selected_identifier is not None and not cameras:
            raise ValueError(f"No Orbbec camera matches '{selected_identifier}'.")

        return cameras

    def get_active_stream_profiles(self) -> dict[str, dict[str, Any] | None]:
        return {
            "color": self.selected_color_stream_profile,
            "depth": self.selected_depth_stream_profile,
        }

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        ob = _import_pyorbbecsdk()
        self.device = self._resolve_device(ob)
        self.device_info = self.device.get_device_info()

        if self.config.depth_work_mode is not None:
            self._set_depth_work_mode(self.config.depth_work_mode)

        self.pipeline = ob.Pipeline(self.device)
        self.pipeline_config = ob.Config()

        color_profile = self._select_color_profile(ob)
        self.pipeline_config.enable_stream(color_profile)
        self.selected_color_stream_profile = self._video_profile_to_dict(color_profile)

        if self.config.use_depth:
            depth_profile = self._select_depth_profile(ob, color_profile)
            self.pipeline_config.enable_stream(depth_profile)
            self.selected_depth_stream_profile = self._video_profile_to_dict(depth_profile)
            if self.config.align_depth_to_color:
                align_mode = getattr(ob.OBAlignMode, _ALIGN_MODE_NAMES[self.config.align_mode])
                self.pipeline_config.set_align_mode(align_mode)
            aggregate_mode = getattr(
                ob.OBFrameAggregateOutputMode,
                _FRAME_AGGREGATE_OUTPUT_MODE_NAMES[self.config.frame_aggregate_output_mode],
            )
            self.pipeline_config.set_frame_aggregate_output_mode(aggregate_mode)
        else:
            self.selected_depth_stream_profile = None

        pipeline_started = False
        try:
            self.pipeline.start(self.pipeline_config)
            pipeline_started = True
            if self.config.use_depth:
                self._apply_depth_tuning_properties(ob)
        except Exception as exc:
            if pipeline_started and self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            self.pipeline = None
            self.pipeline_config = None
            self.selected_color_stream_profile = None
            self.selected_depth_stream_profile = None
            raise ConnectionError(f"Failed to start or configure {self}: {exc}") from exc

        self._apply_profile(color_profile)

        if warmup:
            self._warmup_camera()

        logger.info("%s connected.", self)

    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        frames = self._wait_for_frames(timeout_ms, require_color=True)
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError(f"{self} did not receive a color frame.")

        image = self._color_frame_to_rgb(color_frame)
        return self._postprocess_color_image(image, color_mode)

    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.config.use_depth:
            raise RuntimeError(f"{self} depth stream is not enabled.")

        frames = self._wait_for_frames(timeout_ms, require_depth=True)
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            raise RuntimeError(f"{self} did not receive a depth frame.")

        depth_map = self._depth_frame_to_array(depth_frame)
        return self._apply_rotation(depth_map)

    def read_rgbd(
        self,
        color_mode: ColorMode | None = None,
        timeout_ms: int = 200,
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.config.use_depth:
            raise RuntimeError(f"{self} depth stream is not enabled.")

        frames = self._wait_for_frames(timeout_ms, require_color=True, require_depth=True)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None:
            raise RuntimeError(f"{self} did not receive a color frame.")
        if depth_frame is None:
            raise RuntimeError(f"{self} did not receive a depth frame.")

        color_image = self._color_frame_to_rgb(color_frame)
        depth_map = self._depth_frame_to_array(depth_frame)
        return self._postprocess_color_image(color_image, color_mode), self._apply_rotation(depth_map)

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: event set but no frame available for {self}.")

        return frame

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"Attempted to disconnect {self}, but it is already disconnected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

        self.pipeline_config = None
        self.device = None
        self.device_info = None
        self.selected_color_stream_profile = None
        self.selected_depth_stream_profile = None
        logger.info("%s disconnected.", self)

    def _resolve_device(self, ob: Any) -> Any:
        context = ob.Context()
        device_list = context.query_devices()
        available: list[tuple[str, str, str]] = []
        selected_identifier = self.config.serial_number_or_name

        for index in range(device_list.get_count()):
            device = device_list.get_device_by_index(index)
            info = device.get_device_info()
            serial_number = info.get_serial_number()
            name = info.get_name()
            uid = info.get_uid()
            available.append((serial_number, name, uid))
            if selected_identifier is not None and selected_identifier in {serial_number, name, uid}:
                return device

        if selected_identifier is not None:
            raise ValueError(
                f"No Orbbec camera matches '{selected_identifier}'. Available devices: {available}"
            )

        if len(available) == 1:
            return device_list.get_device_by_index(0)

        raise ValueError(
            "Multiple Orbbec cameras detected. Set `serial_number_or_name` explicitly. "
            f"Available devices: {available}"
        )

    def _select_color_profile(self, ob: Any) -> Any:
        if self.pipeline is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before selecting a color profile.")

        sensor_type = ob.OBSensorType.COLOR_SENSOR
        default_profile, all_profiles = self._get_video_profiles(sensor_type)
        target_width = self.capture_width or default_profile.get_width()
        target_height = self.capture_height or default_profile.get_height()
        target_fps = self.fps or default_profile.get_fps()
        target_format = self._resolve_format(ob, self.config.color_stream_format) or default_profile.get_format()

        matched_profile = self._select_video_profile(
            profiles=all_profiles,
            strategy=self.config.profile_selection_strategy,
            target_width=target_width,
            target_height=target_height,
            target_fps=target_fps,
            target_format=target_format,
        )
        if matched_profile is None:
            available = self._format_video_profiles(all_profiles)
            raise ConnectionError(
                "No compatible Orbbec color stream profile found for "
                f"width={target_width}, height={target_height}, fps={target_fps}, "
                f"format={_enum_name(target_format)} with strategy='{self.config.profile_selection_strategy}'. "
                f"Available profiles: {available}"
            )
        if (
            self.config.profile_selection_strategy == "closest"
            and not self._match_video_profile(
                profiles=[matched_profile],
                target_width=target_width,
                target_height=target_height,
                target_fps=target_fps,
                target_format=target_format,
            )
        ):
            logger.warning(
                "%s selected closest color profile %s for requested width=%s height=%s fps=%s format=%s.",
                self,
                self._video_profile_to_dict(matched_profile),
                target_width,
                target_height,
                target_fps,
                _enum_name(target_format),
            )
        return matched_profile

    def _select_depth_profile(self, ob: Any, color_profile: Any) -> Any:
        sensor_type = ob.OBSensorType.DEPTH_SENSOR
        default_profile, all_profiles = self._get_video_profiles(sensor_type)
        if self.capture_width is None or self.capture_height is None or self.fps is None:
            target_width = color_profile.get_width()
            target_height = color_profile.get_height()
            target_fps = color_profile.get_fps()
        else:
            target_width = self.capture_width
            target_height = self.capture_height
            target_fps = self.fps
        target_format = self._resolve_format(ob, self.config.depth_stream_format) or default_profile.get_format()

        matched_profile = self._select_video_profile(
            profiles=all_profiles,
            strategy=self.config.profile_selection_strategy,
            target_width=target_width,
            target_height=target_height,
            target_fps=target_fps,
            target_format=target_format,
        )
        if matched_profile is not None:
            if (
                self.config.profile_selection_strategy == "closest"
                and not self._match_video_profile(
                    profiles=[matched_profile],
                    target_width=target_width,
                    target_height=target_height,
                    target_fps=target_fps,
                    target_format=target_format,
                )
            ):
                logger.warning(
                    "%s selected closest depth profile %s for requested width=%s height=%s fps=%s format=%s.",
                    self,
                    self._video_profile_to_dict(matched_profile),
                    target_width,
                    target_height,
                    target_fps,
                    _enum_name(target_format),
                )
            return matched_profile

        if not self.config.align_depth_to_color:
            return default_profile

        available = self._format_video_profiles(all_profiles)
        raise ConnectionError(
            "No compatible Orbbec depth stream profile found for "
            f"width={target_width}, height={target_height}, fps={target_fps}, "
            f"format={_enum_name(target_format)} with strategy='{self.config.profile_selection_strategy}' "
            "while depth alignment is enabled. "
            f"Available profiles: {available}"
        )

    def _get_video_profiles(self, sensor_type: Any) -> tuple[Any, list[Any]]:
        if self.pipeline is None:
            raise RuntimeError(f"{self}: pipeline must be initialized before listing profiles.")

        profile_list = self.pipeline.get_stream_profile_list(sensor_type)
        default_profile = profile_list.get_default_video_stream_profile()
        profiles = self._stream_profile_list_to_video_profiles(profile_list)
        return default_profile, profiles

    @staticmethod
    def _list_depth_work_modes_for_device(device: Any) -> tuple[list[str], str | None]:
        if not hasattr(device, "get_depth_work_mode_list"):
            return [], None

        try:
            work_mode_list = device.get_depth_work_mode_list()
        except Exception:
            return [], None

        mode_names: list[str] = []
        for index in range(work_mode_list.get_count()):
            mode = work_mode_list.get_depth_work_mode_by_index(index)
            mode_names.append(Gemini335LCamera._depth_work_mode_name(mode))

        current_mode_name: str | None = None
        if hasattr(device, "get_depth_work_mode"):
            try:
                current_mode_name = Gemini335LCamera._depth_work_mode_name(device.get_depth_work_mode())
            except Exception:
                current_mode_name = None

        return mode_names, current_mode_name

    @staticmethod
    def _depth_work_mode_name(mode: Any) -> str:
        return str(getattr(mode, "name", mode))

    @staticmethod
    def _canonical_depth_mode_name(mode_name: str) -> str:
        return "".join(char for char in mode_name.casefold() if char.isalnum())

    def _set_depth_work_mode(self, mode_name: str) -> None:
        if self.device is None:
            raise RuntimeError(f"{self}: device must be initialized before setting depth work mode.")
        if not hasattr(self.device, "set_depth_work_mode"):
            raise RuntimeError(f"{self} does not support depth work mode selection through this SDK build.")

        available_modes, _ = self._list_depth_work_modes_for_device(self.device)
        selected_mode_name = mode_name
        if available_modes:
            requested_key = self._canonical_depth_mode_name(mode_name)
            selected_mode_name = ""
            for available_mode in available_modes:
                if self._canonical_depth_mode_name(available_mode) == requested_key:
                    selected_mode_name = available_mode
                    break
        if not selected_mode_name:
            raise ValueError(
                f"{self} does not support depth_work_mode='{mode_name}'. Available modes: {available_modes}"
            )

        try:
            status = self.device.set_depth_work_mode(selected_mode_name)
        except Exception as exc:
            raise RuntimeError(
                f"{self} failed to set depth_work_mode='{selected_mode_name}': {exc}"
            ) from exc

        status_name = _enum_name(status)
        if status_name not in {"STATUS_OK", "OK"}:
            raise RuntimeError(
                f"{self} failed to set depth_work_mode='{selected_mode_name}', SDK status={status_name}."
            )

        verified_mode = self._verify_depth_work_mode(selected_mode_name)
        logger.info("%s set depth work mode to '%s'.", self, verified_mode or selected_mode_name)

    def _verify_depth_work_mode(self, expected_mode_name: str) -> str | None:
        if self.device is None or not hasattr(self.device, "get_depth_work_mode"):
            return None

        try:
            current_mode_name = self._depth_work_mode_name(self.device.get_depth_work_mode())
        except Exception as exc:
            logger.warning("%s set depth_work_mode but readback is unavailable: %s", self, exc)
            return None

        if self._canonical_depth_mode_name(current_mode_name) != self._canonical_depth_mode_name(expected_mode_name):
            raise RuntimeError(
                f"{self} requested depth_work_mode='{expected_mode_name}', but current mode is '{current_mode_name}'."
            )
        return current_mode_name

    def _apply_depth_tuning_properties(self, ob: Any) -> None:
        requested_property_values = {
            "disp_search_range_mode": self.config.disp_search_range_mode,
            "disp_search_offset": self.config.disp_search_offset,
        }

        for option_name, option_value in requested_property_values.items():
            if option_value is None:
                continue

            property_enum_candidates = _DEPTH_INT_PROPERTY_NAMES[option_name]
            property_enum_name = None
            property_id = None
            for enum_name in property_enum_candidates:
                enum_value = getattr(ob.OBPropertyID, enum_name, None)
                if enum_value is not None:
                    property_enum_name = enum_name
                    property_id = enum_value
                    break

            if property_id is None or property_enum_name is None:
                raise RuntimeError(
                    f"{self} SDK does not expose any property in {property_enum_candidates}. "
                    "Upgrade pyorbbecsdk or camera firmware."
                )

            self._set_depth_int_property(
                ob=ob,
                property_id=property_id,
                property_enum_name=property_enum_name,
                option_name=option_name,
                option_value=option_value,
            )

    def _set_depth_int_property(
        self,
        *,
        ob: Any,
        property_id: Any,
        property_enum_name: str,
        option_name: str,
        option_value: int,
    ) -> None:
        if self.device is None:
            raise RuntimeError(f"{self}: device must be initialized before setting depth properties.")
        if not hasattr(self.device, "set_int_property"):
            raise RuntimeError(f"{self} SDK does not support integer device property writes.")

        if hasattr(self.device, "is_property_supported") and hasattr(ob, "OBPermissionType"):
            write_permissions = []
            for permission_name in ("PERMISSION_WRITE", "PERMISSION_READ_WRITE"):
                permission = getattr(ob.OBPermissionType, permission_name, None)
                if permission is not None:
                    write_permissions.append(permission)

            if write_permissions and not any(
                self.device.is_property_supported(property_id, permission) for permission in write_permissions
            ):
                raise RuntimeError(
                    f"{self} does not report write support for property '{property_enum_name}'."
                )

        try:
            self.device.set_int_property(property_id, int(option_value))
        except Exception as exc:
            message = str(exc)
            # errorCode: 2 is commonly returned when firmware/SDK does not truly support
            # the property even though the enum exists.
            if "errorCode: 2" in message:
                firmware_version = None
                sdk_version = None
                try:
                    if self.device_info is not None:
                        firmware_version = self.device_info.get_firmware_version()
                except Exception:
                    firmware_version = None
                try:
                    get_version = getattr(ob, "get_version", None)
                    if callable(get_version):
                        sdk_version = get_version()
                except Exception:
                    sdk_version = None
                raise RuntimeError(
                    f"{self} failed to set {option_name}={option_value} ({property_enum_name}): {exc}. "
                    f"Detected firmware={firmware_version}, sdk={sdk_version}. "
                    "For Gemini 330/335 series, extended disparity search range typically requires "
                    "firmware >= 1.3.25 and SDK >= 1.10.12."
                ) from exc
            raise RuntimeError(
                f"{self} failed to set {option_name}={option_value} ({property_enum_name}): {exc}"
            ) from exc

        readback_value: int | None = None
        if hasattr(self.device, "get_int_property"):
            try:
                readback_value = int(self.device.get_int_property(property_id))
            except Exception as exc:
                logger.warning(
                    "%s set %s=%s (%s), but readback is unavailable on this firmware/SDK: %s",
                    self,
                    option_name,
                    option_value,
                    property_enum_name,
                    exc,
                )

        if readback_value is not None and readback_value != int(option_value):
            raise RuntimeError(
                f"{self} requested {option_name}={option_value} ({property_enum_name}), "
                f"but read back {readback_value}."
            )

        logger.info("%s set %s=%s (%s).", self, option_name, option_value, property_enum_name)

    @staticmethod
    def _get_device_sensor_profiles(device: Any, sensor_type: Any) -> tuple[Any | None, list[Any], str | None]:
        try:
            sensor = device.get_sensor(sensor_type)
            profile_list = sensor.get_stream_profile_list()
            default_profile = profile_list.get_default_video_stream_profile()
            profiles = Gemini335LCamera._stream_profile_list_to_video_profiles(profile_list)
            return default_profile, profiles, None
        except Exception as exc:
            return None, [], str(exc)

    @staticmethod
    def _stream_profile_list_to_video_profiles(profile_list: Any) -> list[Any]:
        profiles: list[Any] = []
        for index in range(profile_list.get_count()):
            profile = profile_list.get_stream_profile_by_index(index)
            if hasattr(profile, "is_video_stream_profile") and profile.is_video_stream_profile():
                profiles.append(profile.as_video_stream_profile())
            else:
                profiles.append(profile)
        return profiles

    @staticmethod
    def _select_video_profile(
        *,
        profiles: list[Any],
        strategy: str,
        target_width: int,
        target_height: int,
        target_fps: int,
        target_format: Any,
    ) -> Any | None:
        if strategy == "closest":
            return Gemini335LCamera._closest_video_profile(
                profiles=profiles,
                target_width=target_width,
                target_height=target_height,
                target_fps=target_fps,
                target_format=target_format,
            )
        return Gemini335LCamera._match_video_profile(
            profiles=profiles,
            target_width=target_width,
            target_height=target_height,
            target_fps=target_fps,
            target_format=target_format,
        )

    @staticmethod
    def _match_video_profile(
        *,
        profiles: list[Any],
        target_width: int,
        target_height: int,
        target_fps: int,
        target_format: Any,
    ) -> Any | None:
        for profile in profiles:
            if (
                profile.get_width() == target_width
                and profile.get_height() == target_height
                and profile.get_fps() == target_fps
                and profile.get_format() == target_format
            ):
                return profile
        return None

    @staticmethod
    def _closest_video_profile(
        *,
        profiles: list[Any],
        target_width: int,
        target_height: int,
        target_fps: int,
        target_format: Any,
    ) -> Any | None:
        candidates = [profile for profile in profiles if profile.get_format() == target_format]
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda profile: (
                Gemini335LCamera._profile_distance_score(
                    profile=profile,
                    target_width=target_width,
                    target_height=target_height,
                    target_fps=target_fps,
                ),
                -profile.get_fps(),
                -(profile.get_width() * profile.get_height()),
            ),
        )

    @staticmethod
    def _profile_distance_score(
        *,
        profile: Any,
        target_width: int,
        target_height: int,
        target_fps: int,
    ) -> int:
        # Width/height deviation matters more than fps when searching for the nearest mode.
        return (
            abs(profile.get_width() - target_width) * 4
            + abs(profile.get_height() - target_height) * 4
            + abs(profile.get_fps() - target_fps) * 100
        )

    @staticmethod
    def _format_video_profiles(profiles: list[Any]) -> list[dict[str, Any]]:
        return [Gemini335LCamera._video_profile_to_dict(profile) for profile in profiles]

    @staticmethod
    def _group_video_profiles(profiles: list[Any]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, int, int], set[int]] = {}
        for profile in profiles:
            format_name = _enum_name(profile.get_format())
            key = (format_name, profile.get_width(), profile.get_height())
            grouped.setdefault(key, set()).add(profile.get_fps())

        grouped_profiles = [
            {
                "format": format_name,
                "width": width,
                "height": height,
                "fps": sorted(fps_values),
            }
            for (format_name, width, height), fps_values in grouped.items()
        ]
        grouped_profiles.sort(
            key=lambda item: (item["format"], item["width"] * item["height"], item["width"], item["height"])
        )
        return grouped_profiles

    @staticmethod
    def _video_profile_to_dict(profile: Any) -> dict[str, Any]:
        return {
            "width": profile.get_width(),
            "height": profile.get_height(),
            "fps": profile.get_fps(),
            "format": _enum_name(profile.get_format()),
        }

    @staticmethod
    def _resolve_format(ob: Any, format_name: str | None) -> Any | None:
        if format_name is None:
            return None
        try:
            return getattr(ob.OBFormat, format_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown Orbbec stream format '{format_name}'.") from exc

    def _apply_profile(self, color_profile: Any) -> None:
        actual_width = color_profile.get_width()
        actual_height = color_profile.get_height()
        self.fps = color_profile.get_fps()

        if self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            self.width = actual_height
            self.height = actual_width
            self.capture_width = actual_width
            self.capture_height = actual_height
        else:
            self.width = actual_width
            self.height = actual_height
            self.capture_width = actual_width
            self.capture_height = actual_height

    def _warmup_camera(self) -> None:
        deadline = time.monotonic() + self.warmup_s
        while time.monotonic() < deadline:
            try:
                self.read(timeout_ms=200)
            except TimeoutError:
                logger.debug("%s warmup is still waiting for the first frame.", self)
            time.sleep(0.1)

    def _wait_for_frames(
        self,
        timeout_ms: int,
        *,
        require_color: bool = False,
        require_depth: bool = False,
    ) -> Any:
        if self.pipeline is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        deadline = time.monotonic() + (timeout_ms / 1000.0)
        last_missing: str | None = None

        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                break

            try:
                frames = self.pipeline.wait_for_frames(max(1, remaining_ms))
            except Exception as exc:
                raise RuntimeError(f"{self} failed waiting for frames: {exc}") from exc

            if frames is None:
                last_missing = "frameset"
                continue

            if require_color and frames.get_color_frame() is None:
                last_missing = "color frame"
                continue

            if require_depth and frames.get_depth_frame() is None:
                last_missing = "depth frame"
                continue

            return frames

        if last_missing == "frameset":
            raise TimeoutError(f"{self} did not receive any frames within {timeout_ms} ms.")
        expected_streams = [
            stream_name
            for enabled, stream_name in (
                (require_color, "color frame"),
                (require_depth, "depth frame"),
            )
            if enabled
        ]
        expected_description = " and ".join(expected_streams) or "frames"
        raise TimeoutError(f"{self} did not receive a complete {expected_description} within {timeout_ms} ms.")

    def _postprocess_color_image(
        self, image: NDArray[Any], color_mode: ColorMode | None = None
    ) -> NDArray[Any]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"{self} returned an invalid color frame shape: {image.shape}")

        if self.capture_width is None or self.capture_height is None:
            raise RuntimeError(f"{self} capture dimensions are not initialized.")

        if image.shape[1] != self.capture_width or image.shape[0] != self.capture_height:
            raise RuntimeError(
                f"{self} frame shape {image.shape[:2]} does not match "
                f"configured capture size {(self.capture_height, self.capture_width)}."
            )

        processed = image
        output_color_mode = color_mode or self.color_mode
        if output_color_mode == ColorMode.BGR:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        elif output_color_mode != ColorMode.RGB:
            raise ValueError(
                f"Invalid requested color mode '{output_color_mode}'. "
                f"Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        return self._apply_rotation(processed)

    def _apply_rotation(self, image: NDArray[Any]) -> NDArray[Any]:
        if self.rotation in (
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ):
            return cv2.rotate(image, self.rotation)
        return image

    def _color_frame_to_rgb(self, color_frame: Any) -> NDArray[Any]:
        ob = _import_pyorbbecsdk()
        frame_format = color_frame.get_format()
        width = color_frame.get_width()
        height = color_frame.get_height()
        data = np.asarray(color_frame.get_data(), dtype=np.uint8)

        if frame_format == ob.OBFormat.RGB:
            return data.reshape(height, width, 3)
        if frame_format == ob.OBFormat.BGR:
            return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_BGR2RGB)
        if frame_format in (ob.OBFormat.YUYV, ob.OBFormat.YUY2):
            return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2RGB_YUY2)
        if frame_format == ob.OBFormat.UYVY:
            return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2RGB_UYVY)
        if frame_format == ob.OBFormat.NV12:
            return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_NV12)
        if frame_format == ob.OBFormat.NV21:
            return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_NV21)
        if frame_format == ob.OBFormat.I420:
            return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGB_I420)
        if frame_format == ob.OBFormat.MJPG:
            decoded = cv2.imdecode(data.reshape(-1), cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError(f"{self} failed to decode MJPG color frame.")
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        if frame_format == ob.OBFormat.BGRA:
            return cv2.cvtColor(data.reshape(height, width, 4), cv2.COLOR_BGRA2RGB)
        if frame_format == ob.OBFormat.RGBA:
            return cv2.cvtColor(data.reshape(height, width, 4), cv2.COLOR_RGBA2RGB)

        raise RuntimeError(f"{self} does not support color frame format {_enum_name(frame_format)}.")

    def _depth_frame_to_array(self, depth_frame: Any) -> NDArray[Any]:
        ob = _import_pyorbbecsdk()
        frame_format = depth_frame.get_format()
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        data = np.asarray(depth_frame.get_data(), dtype=np.uint8)

        if frame_format in (ob.OBFormat.Y16, ob.OBFormat.Z16, ob.OBFormat.RW16):
            depth_map = np.frombuffer(data.tobytes(), dtype=np.uint16)
            if depth_map.size != width * height:
                raise RuntimeError(
                    f"{self} returned an invalid depth buffer size for format {_enum_name(frame_format)}."
                )
            return depth_map.reshape(height, width)
        if frame_format in (ob.OBFormat.Y8, ob.OBFormat.GRAY):
            return data.reshape(height, width)

        pixel_bits = getattr(depth_frame, "get_pixel_available_bit_size", lambda: 0)()
        if pixel_bits == 16 and data.size == width * height * 2:
            return np.frombuffer(data.tobytes(), dtype=np.uint16).reshape(height, width)

        raise RuntimeError(f"{self} does not support depth frame format {_enum_name(frame_format)}.")

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        while not self.stop_event.is_set():
            try:
                frame = self.read(timeout_ms=500)
                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except Exception as exc:
                logger.warning("Error reading frame in background thread for %s: %s", self, exc)

    def _start_read_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None
