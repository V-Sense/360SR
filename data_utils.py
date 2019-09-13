from os import listdir
from os.path import join
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

import torchvision.transforms.functional as TF

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])
    
        
    
def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

# initializing / creating main (temp) directories (aux and debug)
def init_aux(data_dir, aux_dir):
    if not os.path.exists(os.path.join(data_dir,aux_dir)):
        print("creating temp directories at", data_dir)
        os.makedirs(os.path.join(data_dir,aux_dir))

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        self.upscale_factor = upscale_factor
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
#        self.hr_transform = train_hr_transform(crop_size)
##        self.lr_transform = train_lr_transform(crop_size, upscale_factor)
#        self.lr_transform = train_hr_transform(crop_size)
        
    def train_transform(self, hr, lr, crop_size, upscale_factor):
        wd_hr, ht_hr = hr.size
        wd_lr = int(crop_size/upscale_factor)
        ht_lr = wd_lr
        # print(ht_lr,wd_lr)
        i, j, h, w = RandomCrop.get_params(hr, output_size=(crop_size, crop_size))
        hr_train = TF.crop(hr, i, j, h, w)
        if upscale_factor == 1:
            lr_train = TF.crop(lr, i, j, h, w)
        else: 
            lr_temp = lr.resize((wd_hr,ht_hr),Image.BICUBIC)
            lr_train = TF.crop(lr_temp, i, j, h, w)
            lr_train = lr_train.resize((wd_lr,ht_lr),Image.BICUBIC)
        hr_train = TF.to_tensor(hr_train)
        lr_train = TF.to_tensor(lr_train)
        return lr_train, hr_train

    def __getitem__(self, index):
        lr_image = Image.open(self.image_filenames[index])
        path, im_name = os.path.split(self.image_filenames[index])
        hr_image = Image.open(join('data/trans_Equi/train/',im_name))
        # print(self.image_filenames[index])
        lr_image, hr_image = self.train_transform(hr_image, lr_image, 88 , self.upscale_factor)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
