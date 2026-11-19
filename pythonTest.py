import argparse
import json
import os

import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--boxes_dir', type=str, default='boxes')
parser.add_argument('--face_detection_model', '-fd', type=str, required=True)
parser.add_argument('--score_threshold', type=float, default=0.6)
parser.add_argument('--nms_threshold', type=float, default=0.3)
parser.add_argument('--top_k', type=int, default=5000)
args = parser.parse_args()

def make_face_mask(height, width, box):
    x, y, w, h = box
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (int(x + w / 2), int(y + h / 2))
    axes = (max(1, int(w * 0.45)), max(1, int(h * 0.55)))
    cv.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask

def blur_face_oval(img, box):
    height, width = img.shape[:2]
    x, y, w, h = box
    pad = int(0.1 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + w + pad)
    y1 = min(height, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return

    mask = make_face_mask(height, width, box)
    mask_roi = mask[y0:y1, x0:x1]
    roi = img[y0:y1, x0:x1]

    k = max(3, (max(w, h) // 2) | 1)
    blurred = cv.GaussianBlur(roi, (k, k), 0)

    alpha = cv.GaussianBlur(mask_roi, (15, 15), 0).astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]
    blended = blurred.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
    img[y0:y1, x0:x1] = blended.astype(np.uint8)

if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.boxes_dir, exist_ok=True)

    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )

    image_names = [f for f in os.listdir(args.input_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for name in image_names:
        img = cv.imread(os.path.join(args.input_dir, name))
        if img is None:
            print('Could not read {}, skipping'.format(name))
            continue

        height, width = img.shape[:2]
        detector.setInputSize((width, height))
        faces = detector.detect(img)

        boxes = []
        if faces[1] is not None:
            for face in faces[1]:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'score': float(face[-1])})

        base = os.path.splitext(name)[0]
        with open(os.path.join(args.boxes_dir, base + '.json'), 'w') as f:
            json.dump(boxes, f, indent=2)

        for b in boxes:
            blur_face_oval(img, (b['x'], b['y'], b['w'], b['h']))

        cv.imwrite(os.path.join(args.output_dir, name), img)
        print('{}: {} face(s) blurred'.format(name, len(boxes)))

    print('Done.')