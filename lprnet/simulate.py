import cv2
import numpy as np
from rknn.api import RKNN
from decoder import decode, CHARS


DATASET_PATH = './model/dataset.txt'
RKNN_MODEL_PATH = './model/lprnet.rknn'
ONNX_MODEL_PATH = './model/lprnet.onnx'
PLATFORM = "rk3568"

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[
                [127.5, 127.5, 127.5]], target_platform=PLATFORM)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL_PATH)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target=None, device_id=None)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('./model/test.jpg')
    img = cv2.resize(img, (94, 24))

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    # Post Process
    print('--> PostProcess')
    labels, pred_labels = decode(outputs[0], CHARS)
    print('车牌识别结果: ' + labels[0])

    # 量化模型精度分析
    rknn.accuracy_analysis(
        inputs=['./model/test.jpg'],
        output_dir='./accuracy',
        target=None,
        device_id=None,
        # topk=5,
        # accuracy_threshold=0.01, dataset=DATASET_PATH
    )

    # Release
    rknn.release()
