import sys
from rknn.api import RKNN

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

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
