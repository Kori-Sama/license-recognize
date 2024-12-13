import torch
import os
import sys
import urllib
import urllib.request
import time
import traceback
from lprnet import LPRNet

MODEL_DIR = './model/'
MODEL_PATH = MODEL_DIR + 'Final_LPRNet_model.pth'


def readable_speed(speed):
    speed_bytes = float(speed)
    speed_kbytes = speed_bytes / 1024
    if speed_kbytes > 1024:
        speed_mbytes = speed_kbytes / 1024
        if speed_mbytes > 1024:
            speed_gbytes = speed_mbytes / 1024
            return "{:.2f} GB/s".format(speed_gbytes)
        else:
            return "{:.2f} MB/s".format(speed_mbytes)
    else:
        return "{:.2f} KB/s".format(speed_kbytes)


def show_progress(blocknum, blocksize, totalsize):
    speed = (blocknum * blocksize) / (time.time() - start_time)
    speed_str = " Speed: {}".format(readable_speed(speed))
    recv_size = blocknum * blocksize

    f = sys.stdout
    progress = (recv_size / totalsize)
    progress_str = "{:.2f}%".format(progress * 100)
    n = round(progress * 50)
    s = ('#' * n).ljust(50, '-')
    f.write(progress_str.ljust(8, ' ') + '[' + s + ']' + speed_str)
    f.flush()
    f.write('\r\n')


def check_and_download_origin_model():
    global start_time
    if not os.path.exists(MODEL_PATH):
        print('--> Download {}'.format(MODEL_PATH))
        url = 'https://github.com/sirius-ai/LPRNet_Pytorch/raw/master/weights/Final_LPRNet_model.pth'
        download_file = MODEL_PATH
        try:
            start_time = time.time()
            urllib.request.urlretrieve(url, download_file, show_progress)
        except:
            print('Download {} failed.'.format(download_file))
            print(traceback.format_exc())
            exit(-1)
        print('done')


if __name__ == "__main__":

    # Download model if not exist (from https://github.com/sirius-ai/LPRNet_Pytorch/blob/master/weights)
    check_and_download_origin_model()

    device = torch.device('cpu')
    lprnet = LPRNet(class_num=68, dropout_rate=0).to(device)
    lprnet.load_state_dict(torch.load('../model/Final_LPRNet_model.pth'))
    lprnet.eval()

    torch.onnx.export(lprnet,
                      torch.randn(1, 3, 24, 94),
                      MODEL_DIR + 'lprnet.onnx',
                      export_params=True,
                      input_names=['input'],
                      output_names=['output'],
                      )

    if os.path.exists(MODEL_DIR + 'lprnet.onnx'):
        print('onnx model had been saved in ' + MODEL_DIR + 'lprnet.onnx')
    else:
        print('export onnx failed!')
