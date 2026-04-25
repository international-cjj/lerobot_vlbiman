from __future__ import annotations

import argparse
import shlex

from .diagnostics import (
    build_recommended_depth_viewer_command,
    diagnose_depth_startup,
    format_diagnosis_report,
    save_report_json,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Gemini 335L RGB/RGBD startup health and print a recommended "
            "depth_viewer command that is most likely to work."
        )
    )
    parser.add_argument("--serial-number-or-name", default=None, help="Preferred camera serial/name/uid.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--color-stream-format", type=str, default="MJPG")
    parser.add_argument("--depth-stream-format", type=str, default="Y16")
    parser.add_argument("--depth-work-mode", type=str, default=None)
    parser.add_argument("--disp-search-range-mode", type=int, default=None)
    parser.add_argument("--disp-search-offset", type=int, default=None)
    parser.add_argument(
        "--profile-selection-strategy",
        choices=("exact", "closest"),
        default="exact",
        help="exact: strict mode matching; closest: fallback to nearest available mode.",
    )
    parser.add_argument("--align-depth-to-color", action="store_true")
    parser.add_argument(
        "--align-mode",
        choices=("hw", "sw"),
        default="sw",
        help="Use 'sw' for better compatibility during diagnostics.",
    )
    parser.add_argument("--read-count", type=int, default=15, help="Frame reads per startup probe.")
    parser.add_argument("--timeout-ms", type=int, default=700)
    parser.add_argument(
        "--probe-all-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, continue probing other cameras when the preferred one fails.",
    )
    parser.add_argument(
        "--stop-on-first-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, stop as soon as an RGBD configuration succeeds.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/gemini335l_diagnostics/depth_startup_report.json",
        help="Path to write a machine-readable JSON report.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    report = diagnose_depth_startup(
        preferred_identifier=args.serial_number_or_name,
        requested={
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "color_stream_format": args.color_stream_format,
            "depth_stream_format": args.depth_stream_format,
            "depth_work_mode": args.depth_work_mode,
            "disp_search_range_mode": args.disp_search_range_mode,
            "disp_search_offset": args.disp_search_offset,
            "align_depth_to_color": args.align_depth_to_color,
            "align_mode": args.align_mode,
            "profile_selection_strategy": args.profile_selection_strategy,
        },
        read_count=args.read_count,
        timeout_ms=args.timeout_ms,
        probe_all_cameras=args.probe_all_cameras,
        stop_on_first_success=args.stop_on_first_success,
    )

    print(format_diagnosis_report(report))

    json_path = save_report_json(report, args.output_json)
    print(f"\nSaved report JSON: {json_path}")

    recommended_command = build_recommended_depth_viewer_command(report)
    if recommended_command:
        print(f"\nCOPY-PASTE (current env): PYTHONPATH=src:. {shlex.join(recommended_command)}")
        return

    raise SystemExit(1)


if __name__ == "__main__":
    main()
