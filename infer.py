from yolo import onnx_infer
from lprnet import rknn_infer
import cv2

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
YOLO_MODEL = 'yolo/model/best.onnx'
LPRNET_MODEL = 'lprnet/model/lprnet.onnx'


def detect(model, img):
    detection = onnx_infer.YOLOv8(
        model, img, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
    return detection.detect_box_img()


def recognize(model, img):
    return rknn_infer.simulate_recognize(model, img)


def draw(orign_img, box):
    x1, y1, w, h = box
    color = (0, 255, 0)
    cv2.rectangle(orign_img, (int(x1), int(y1)),
                  (int(x1 + w), int(y1 + h)), color, 2)

    return orign_img


def run(img):
    cropped = detect(YOLO_MODEL, img)
    cropped_img = cropped["img"]
    box = cropped["box"]
    score = cropped["score"]
    label = recognize(LPRNET_MODEL, cropped_img)

    output_img = draw(cv2.imread(img), box)

    return output_img, label, score
