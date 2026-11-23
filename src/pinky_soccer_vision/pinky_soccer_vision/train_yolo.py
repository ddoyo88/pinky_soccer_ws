#!/usr/bin/env python3
"""Train a YOLO golf-ball detector.

  pip install ultralytics --break-system-packages        # or in a venv
  ros2 run pinky_soccer_vision train_yolo --data ~/datasets/golfball/data.yaml --epochs 60

Then run the game with:  backend:=yolo model_path:=<printed best.pt path>
"""
import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='data.yaml written by collect_dataset')
    ap.add_argument('--model', default='yolo11n.pt', help='pretrained base (n = fastest, fine for 1 class)')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default=None, help="e.g. 0 for the first GPU, 'cpu'")
    ap.add_argument('--project', default=os.path.expanduser('~/pinky_soccer_runs'))
    ap.add_argument('--export-onnx', action='store_true')
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=a.project, name='golfball', exist_ok=True,
                # the ball is tiny: keep mosaic, add mild colour jitter, no flips that would change nothing useful
                hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=0.0, fliplr=0.5, mosaic=1.0)
    best = os.path.join(a.project, 'golfball', 'weights', 'best.pt')
    print('\nbest weights:', best)
    if a.export_onnx:
        YOLO(best).export(format='onnx', imgsz=a.imgsz)
    dst = os.path.join(a.project, 'ball.pt')
    if os.path.exists(best):
        shutil.copy(best, dst)
        print('copied to', dst)


if __name__ == '__main__':
    main()
