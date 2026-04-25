from dataclasses import dataclass

from lerobot.cameras import CameraConfig
from lerobot.cameras.configs import ColorMode, Cv2Rotation

_VALID_ALIGN_MODES = {"hw", "sw"}
_VALID_FRAME_AGGREGATE_OUTPUT_MODES = {
    "full_frame_require",
    "color_frame_require",
    "any_situation",
}
_VALID_PROFILE_SELECTION_STRATEGIES = {
    "exact",
    "closest",
}


@CameraConfig.register_subclass("gemini335l")
@dataclass
class Gemini335LCameraConfig(CameraConfig):
    serial_number_or_name: str | None = None
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    use_depth: bool = False
    align_depth_to_color: bool = False
    align_mode: str = "hw"
    frame_aggregate_output_mode: str = "full_frame_require"
    color_stream_format: str | None = None
    depth_stream_format: str | None = None
    profile_selection_strategy: str = "exact"
    depth_work_mode: str | None = None
    disp_search_range_mode: int | None = None
    disp_search_offset: int | None = None

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, "
                f"but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                "`rotation` is expected to be one of "
                f"{(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, "
                f"but {self.rotation} is provided."
            )

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )

        if self.align_depth_to_color and not self.use_depth:
            raise ValueError("`align_depth_to_color=True` requires `use_depth=True`.")

        self.align_mode = self.align_mode.lower()
        if self.align_mode not in _VALID_ALIGN_MODES:
            raise ValueError(
                f"`align_mode` must be one of {sorted(_VALID_ALIGN_MODES)}, but '{self.align_mode}' is provided."
            )

        self.frame_aggregate_output_mode = self.frame_aggregate_output_mode.lower()
        if self.frame_aggregate_output_mode not in _VALID_FRAME_AGGREGATE_OUTPUT_MODES:
            raise ValueError(
                "`frame_aggregate_output_mode` must be one of "
                f"{sorted(_VALID_FRAME_AGGREGATE_OUTPUT_MODES)}, "
                f"but '{self.frame_aggregate_output_mode}' is provided."
            )

        self.profile_selection_strategy = self.profile_selection_strategy.lower()
        if self.profile_selection_strategy not in _VALID_PROFILE_SELECTION_STRATEGIES:
            raise ValueError(
                "`profile_selection_strategy` must be one of "
                f"{sorted(_VALID_PROFILE_SELECTION_STRATEGIES)}, "
                f"but '{self.profile_selection_strategy}' is provided."
            )

        self.color_stream_format = self._normalize_format(self.color_stream_format, "color_stream_format")
        self.depth_stream_format = self._normalize_format(self.depth_stream_format, "depth_stream_format")
        self.depth_work_mode = self._normalize_mode_name(self.depth_work_mode)
        self.disp_search_range_mode = self._normalize_optional_int(
            self.disp_search_range_mode, "disp_search_range_mode"
        )
        self.disp_search_offset = self._normalize_optional_int(self.disp_search_offset, "disp_search_offset")

        if not self.use_depth and self.depth_stream_format is not None:
            raise ValueError("`depth_stream_format` requires `use_depth=True`.")
        if not self.use_depth and self.disp_search_range_mode is not None:
            raise ValueError("`disp_search_range_mode` requires `use_depth=True`.")
        if not self.use_depth and self.disp_search_offset is not None:
            raise ValueError("`disp_search_offset` requires `use_depth=True`.")

    @staticmethod
    def _normalize_format(value: str | None, field_name: str) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError(f"`{field_name}` must not be empty if provided.")
        return normalized

    @staticmethod
    def _normalize_mode_name(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("`depth_work_mode` must not be empty if provided.")
        return normalized

    @staticmethod
    def _normalize_optional_int(value: int | str | None, field_name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"`{field_name}` must be an integer, but bool is provided.")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError(f"`{field_name}` must not be empty if provided.")
            try:
                return int(normalized)
            except ValueError as exc:
                raise ValueError(f"`{field_name}` must be an integer, but '{value}' is provided.") from exc
        raise ValueError(f"`{field_name}` must be an integer, but '{type(value).__name__}' is provided.")
