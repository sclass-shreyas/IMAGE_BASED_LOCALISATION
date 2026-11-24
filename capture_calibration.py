"""Interactive ground markers: freeze, click, type right,forward metres, save."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from src.video_source import VideoSource, add_video_arguments, config_from_arguments
from src.ground_calibration import fit_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_video_arguments(parser)
    parser.add_argument("--output", default="ground_points.json")
    args = parser.parse_args()
    config = config_from_arguments(args)
    source = VideoSource(config).start()
    frozen, latest = None, None
    points, pending = [], []
    text, message, validation = "", "Space: freeze. Q: quit.", False
    window = "Ground calibration"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and frozen is not None and not pending:
            pending.append([x, y])

    cv2.setMouseCallback(window, click)
    try:
        while True:
            packet = source.read(0.03)
            if packet is not None:
                latest = packet["frame"]
            image = frozen if frozen is not None else latest
            if image is None:
                image = np.zeros((480, 900, 3), np.uint8)
                message = str(source.status())
            display = image.copy()
            for i, point in enumerate(points):
                xy = tuple(map(int, point["pixel_xy"]))
                cv2.circle(display, xy, 5, (0, 200, 255), -1)
                cv2.putText(display, f"{i+1}:{point['ground_right_forward_m']}", xy,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if pending:
                cv2.circle(display, tuple(pending[0]), 6, (0, 0, 255), -1)
            prompt = f"Type right,forward metres: {text}" if pending else message
            cv2.rectangle(display, (0, 0), (display.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(display, prompt[:140], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display, f"Space freeze | U undo | R restart | V validation={validation} | S save | Q quit",
                        (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(10) & 0xff
            if key == ord('q') or key == 27:
                break
            if key == ord('r'):
                frozen, points, pending, text = None, [], [], ""
                continue
            if key == ord('u'):
                if pending:
                    pending.clear()
                    text = ""
                elif points:
                    points.pop()
                continue
            if pending:
                if key in (10, 13):
                    try:
                        coords = [float(v.strip()) for v in text.split(',')]
                        if len(coords) != 2 or not np.isfinite(coords).all():
                            raise ValueError()
                        points.append(dict(pixel_xy=pending.pop(), ground_right_forward_m=coords,
                                           validation=validation))
                        text, message = "", "Click another measured marker, or S to save."
                    except ValueError:
                        text = ""
                elif key in (8, 127):
                    text = text[:-1]
                elif key < 128 and chr(key) in "0123456789.,-+ eE":
                    text += chr(key)
                continue
            if key == 32 and latest is not None and frozen is None:
                frozen = latest.copy()
                message = "Click a marker, type right,forward coordinates, press Enter."
            elif key == ord('v'):
                validation = not validation
            elif key == ord('s') and frozen is not None:
                data = dict(image_width=frozen.shape[1], image_height=frozen.shape[0],
                            source_description=f"{config.source_type}; {frozen.shape[1]}x{frozen.shape[0]}", points=points)
                try:
                    result = fit_calibration(data)
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")
                    cv2.imwrite(str(Path(args.output).with_suffix('.png')), frozen)
                    message = f"Saved {args.output}; RMS {result['rms_error_m']:.3f} m"
                    print(message)
                except ValueError as exc:
                    message = str(exc)
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
