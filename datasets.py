import os
import os.path
import numpy as np
import random
import torch
import cv2
import torch.utils.data as data
import h5py
from PIL import Image
import torchvision.transforms as transforms
from random import choice
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def normalize(data):
    return data / 255.

class AllweatherData(data.Dataset):
    def __init__(self, crop_size=(256, 256), train_data_dir='./dataset/AllinOne/', train_filename='allweather.txt'):
        super().__init__()
        train_list = train_data_dir + train_filename
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input', 'gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir
        self.train_filename = train_filename

    def __getitem__(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        if self.train_filename == 'raindroptesta.txt':
            gt_name = gt_name.replace('rain.', 'clean.')
        if self.train_filename == 'test1.txt':
            gt_name = gt_name[:18] + '.png'
        img_path = os.path.join(self.train_data_dir, input_name)
        gt_path = os.path.join(self.train_data_dir, gt_name)
        input = cv2.imread(img_path)
        input = cv2.resize(input, (256, 256))
        b, g, r = cv2.split(input)
        input = cv2.merge([r, g, b])
        input = np.float32(normalize(input))
        input = input.transpose(2, 0, 1)
        gt = cv2.imread(gt_path)
        gt = cv2.resize(gt, (256, 256))
        b, g, r = cv2.split(gt)
        gt = cv2.merge([r, g, b])
        gt = np.float32(normalize(gt))
        gt = gt.transpose(2, 0, 1)

        return torch.Tensor(input), torch.Tensor(gt)

    def __len__(self):
        return len(self.input_names)



class spaDataset(data.Dataset):
    def __init__(self, data_path='./Real_Rain_Streaks_Dataset_CVPR19/', train=True):
        self.train = train
        if self.train:
            self.data_path = os.path.join(data_path, 'Training')
            with open(os.path.join(self.data_path, 'real_world.txt'), 'r') as f:
                self.data_lis = f.readlines()


        else:
            self.data_path = os.path.join(data_path, 'Testing', 'real_test_1000')
            self.data_lis = os.listdir(os.path.join(self.data_path, 'rain'))

    def __len__(self):
            return len(self.data_lis)

    def __getitem__(self, index):
        if self.train:
            line = self.data_lis[index]
            img_name, gt_name = line.strip('\n').split(' ')
            img_path = self.data_path + img_name
            gt_path = self.data_path + gt_name

        else:
            img_name = self.data_lis[index]
            img_path = os.path.join(self.data_path, 'rain', img_name)
            gt_name = img_name.split('.')[0]+'gt'+'.png'
            gt_path = os.path.join(self.data_path, 'gt', gt_name)

        input = cv2.imread(img_path)
        input = cv2.resize(input, (256, 256))
        b, g, r = cv2.split(input)
        input = cv2.merge([r, g, b])
        input = np.float32(normalize(input))
        input = input.transpose(2, 0, 1)
        gt = cv2.imread(gt_path)
        gt = cv2.resize(gt, (256, 256))
        b, g, r = cv2.split(gt)
        gt = cv2.merge([r, g, b])
        gt = np.float32(normalize(gt))
        gt = gt.transpose(2, 0, 1)

        return torch.Tensor(input), torch.Tensor(gt)




class rainDataset(data.Dataset):
    def __init__(self, data_path='.', train=True):
        super(rainDataset, self).__init__()
        self.train = train
        self.data_path = data_path

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        if self.train:
            target_path = os.path.join(self.data_path, 'train_target.h5')
            target_h5f = h5py.File(target_path, 'r')
            target = np.array(target_h5f[key])
            target_h5f.close()
        else:
            target = [0]

        input_path = os.path.join(self.data_path, 'train_input.h5')
        input_h5f = h5py.File(input_path, 'r')
        input = np.array(input_h5f[key])
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)


class cityDataset(data.Dataset):
    def __init__(self, data_path='.', train='train', gt_path='.'):
        self.train = train
        if self.train != 'test':
            self.data_path = os.path.join(data_path, 'train')
        elif self.train == 'test':
            self.data_path = os.path.join(data_path, 'val')

        self.data_lis = []
        for lis in os.listdir(self.data_path):
            self.data_lis.extend(os.listdir(os.path.join(self.data_path, lis)))
        # self.gt_path = gt_path
        if self.train == 'train' or self.train == 'val':
            self.gt_path = os.path.join(gt_path, 'train')
        elif self.train == 'test':
            self.gt_path = os.path.join(gt_path, 'val')

    def __len__(self):
        return len(self.data_lis)

    def __getitem__(self, index):
        img_name = self.data_lis[index]
        lis_name = img_name.split('_')[0]
        img_path = os.path.join(self.data_path, lis_name)
        input = cv2.imread(os.path.join(img_path, img_name))
        h, w = input.shape[:2]
        input = cv2.resize(input, (w // 4, h // 4))
        b, g, r = cv2.split(input)
        input = cv2.merge([r, g, b])
        input = np.float32(normalize(input))
        input = input.transpose(2, 0, 1)

        gt_path = os.path.join(self.gt_path, lis_name)
        gt_name = '_'.join(img_name.split('_')[0:4]) + '.png'
        gt = cv2.imread(os.path.join(gt_path, gt_name))
        gt = cv2.resize(gt, (w // 4, h // 4))  # downsample
        b, g, r = cv2.split(gt)
        gt = cv2.merge([r, g, b])
        gt = np.float32(normalize(gt))
        gt = gt.transpose(2, 0, 1)

        return torch.Tensor(input), torch.Tensor(gt)

class RESIDEDataset(data.Dataset):
    def __init__(self, data_path='.', train='train', gt_path='.'):
        self.train = train
        self.data_path = data_path
        self.data_lis = os.listdir(data_path)
        self.gt_path = gt_path

    def __len__(self):
        return len(self.data_lis)

    def __getitem__(self, index):
        img_name = self.data_lis[index]
        input = cv2.imread(os.path.join(self.data_path, img_name))
        h, w = input.shape[:2]
        input = cv2.resize(input, (256, 256))
        input = np.float32(normalize(input))
        input = input.transpose(2, 0, 1)


        gt_name = img_name.split('_')[0] + '.png'
        gt = cv2.imread(os.path.join(self.gt_path, gt_name))
        gt = cv2.resize(gt, (256, 256))  # downsample
        gt = np.float32(normalize(gt))
        gt = gt.transpose(2, 0, 1)

        return torch.Tensor(input), torch.Tensor(gt)


class depthDataset(data.Dataset):
    def __init__(self, data_path='.', train='train', gt_path='.', depth_path='.'):
        self.train = train
        if self.train != 'test':
            self.data_path = os.path.join(data_path, 'train')
        elif self.train == 'test':
            self.data_path = os.path.join(data_path, 'val')

        self.data_lis = []
        for lis in os.listdir(self.data_path):
            self.data_lis.extend(os.listdir(os.path.join(self.data_path, lis)))

        if self.train == 'train' or self.train == 'val':
            self.gt_path = os.path.join(gt_path, 'train')
            self.depth_path = os.path.join(depth_path, 'train')
        elif self.train == 'test':
            self.gt_path = os.path.join(gt_path, 'val')
            self.depth_path = os.path.join(depth_path, 'val')
        random.shuffle(self.data_lis)

    def __len__(self):
        return len(self.data_lis)

    def __getitem__(self, index):
        img_name = self.data_lis[index]
        lis_name = img_name.split('_')[0]
        img_path = os.path.join(self.data_path, lis_name)
        input = cv2.imread(os.path.join(img_path, img_name))
        h, w = input.shape[:2]
        input = cv2.resize(input, (w // 4, h // 4))
        b, g, r = cv2.split(input)
        input = cv2.merge([r, g, b])
        input = np.float32(normalize(input))
        input = input.transpose(2, 0, 1)

        # if self.train != 'test':
        gt_path = os.path.join(self.gt_path, lis_name)
        gt_name = '_'.join(img_name.split('_')[0:4]) + '.png'
        gt = cv2.imread(os.path.join(gt_path, gt_name))
        # print(os.path.join(gt_path, gt_name))
        gt = cv2.resize(gt, (w // 4, h // 4))  # downsample
        b, g, r = cv2.split(gt)
        gt = cv2.merge([r, g, b])
        gt = np.float32(normalize(gt))
        gt = gt.transpose(2, 0, 1)

        depth_path = os.path.join(self.depth_path, lis_name)
        depth_name = '_'.join(img_name.split('_')[0:3]) + '_depth_rain.png'
        depth = cv2.imread(os.path.join(depth_path, depth_name))
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        depth = cv2.resize(depth, (128, 64))
        depth = np.float32(normalize(depth))
        depth = depth.reshape(-1, 64, 128)

        return torch.Tensor(input), torch.Tensor(gt), torch.Tensor(depth)



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


if __name__ == '__main__':
    #     print('Loading dataset ...\n')
    #     prepare_data_RainTrainH(data_path=opt.data_path, patch_size=128, stride=100)
    # dataset_train = rainDataset(data_path='./Dataset/rain/RainTrainL')
    # dataset_train = RAINDataset(phase='train')
    # print(dataset_train[2])
    dataset_test = spaDataset(train=False)
    # loader_train = data.DataLoader(dataset=dataset_train, num_workers=4, batch_size=4, shuffle=True)
    loader_test = data.DataLoader(dataset=dataset_test, num_workers=4, batch_size=4, shuffle=True)
    # print(len(dataset_test))
    # print(len(dataset_train))
    #     print("# of training samples: %d\n" % int(len(dataset_train)))
    # print(len(loader_train))
    # for iteration, (a, gt) in enumerate(loader_train):
        # print(a.shape,gt.shape)

    # prepare_data_Raincity(data_path='./leftImg8bit_trainval_rain/leftImg8bit_rain/train',
    #                       gt_path='./leftImg8bit_trainvaltest/leftImg8bit/train/',
    #                       save_path='./leftImg8bit_trainval_rain/leftImg8bit_rain')
