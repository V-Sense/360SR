import argparse
import time
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from data_utils import init_aux
from glob import glob
from model import Generator

parser = argparse.ArgumentParser(description='Test for images')
parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
# parser.add_argument('--folder_name', default = 'data/' + str(UPSCALE_FACTOR) + '/test/', type=str, help='folder name for images')
parser.add_argument('--model_name', default='8x_G.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
init_aux('results', str(UPSCALE_FACTOR) + 'x')
FOLDER = 'data/' + str(UPSCALE_FACTOR) + '/test/'
files_list = glob(os.path.join(FOLDER, '*.png'))
MODEL_NAME = opt.model_name
TEST_MODE = True if opt.test_mode == 'GPU' else False
model = Generator(UPSCALE_FACTOR).eval()

if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('models/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('models/' + MODEL_NAME, map_location=lambda storage, loc: storage))

for a_file in sorted(files_list):
    path, IMAGE_NAME = os.path.split(a_file) 
    image = Image.open(path+'/' +IMAGE_NAME)
    image = Variable(ToTensor()(image), volatile=True).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()
    start = time.clock()
    out = model(image)
    elapsed = (time.clock() - start)
    print('cost' + str(elapsed) + 's')
    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save('results/' + str(UPSCALE_FACTOR) + 'x/' + IMAGE_NAME)
