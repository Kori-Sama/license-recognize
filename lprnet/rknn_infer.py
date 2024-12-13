import cv2
from rknn.api import RKNN
from lprnet.decoder import decode, CHARS

PLATFORM = 'rk3568'
DATASET_PATH = 'lprnet/model/dataset.txt'


def recognize(model, img):
    target = 'rk3568'
    # target = None
    device_id = None

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # Load RKNN model
    ret = rknn.load_rknn(model)
    if ret != 0:
        print('Load RKNN model \"{}\" failed!'.format(model))
        exit(ret)

    print(target)

    ret = rknn.init_runtime(target=target, device_id=device_id)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)

    img = cv2.resize(img, (94, 24))

    outputs = rknn.inference(inputs=[img])

    labels, pred_labels = decode(outputs[0], CHARS)
    # print('车牌识别结果: ' + labels[0])

    # Release
    rknn.release()

    return labels[0]


def simulate_recognize(model, img):
    rknn = RKNN(verbose=False)

    rknn.config(mean_values=[[127.5, 127.5, 127.5]], std_values=[
                [127.5, 127.5, 127.5]], target_platform=PLATFORM)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model)
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

    img = cv2.resize(img, (94, 24))

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    # Post Process
    print('--> PostProcess')
    labels, pred_labels = decode(outputs[0], CHARS)

    return labels[0]


if __name__ == '__main__':
    model_path = './model/lprnet.rknn'
