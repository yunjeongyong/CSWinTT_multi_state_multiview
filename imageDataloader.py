from __future__ import annotations

import os
from PIL import Image
import numpy as np
import cv2
import torch


class Load2DFolder(torch.utils.data.Dataset):
    def __init__(self, root, main_camera=0, frame_range=None, focal_range=None):
        self.root = root
        self.main_camera = main_camera
        # self.frames = [str(i).zfill(3) for i in range(frame_range[0], frame_range[1] + 1)]
        self.paths = [os.path.join(self.root, i) for i in os.listdir(self.root)]
        self.paths.sort()
        self.files: list[list[str]] = []
        self.seven_files: list[list[str]] = []
        self.sum_files: list[list[str]] = []
        self.focal_range = focal_range
        self.frame_range = frame_range
        self.iter_idx = 0

        for i, path in enumerate(self.paths):
            # h_000_000.jpg
            a = path.split("\\")
            path = a[-1]
            _, frame, camera = path.split("_")
            camera = camera.split(".")[0]

            if int(self.frame_range[0]) <= int(frame) <= int(self.frame_range[1]):
                if self.focal_range is None:
                    if int(camera) == 0:
                        self.sum_files.append([])
                        self.files.append([])
                        if len(self.files) < int(frame):
                            for _ in range(0, int(frame) - len(self.files) + 1):
                                self.files.append([])
                                self.sum_files.append([])
                        # print("frame: %d, files: %d" % (int(frame), len(self.files)))
                        self.files[int(frame)].append(os.path.join(self.root, path))
                        self.sum_files[int(frame)].append(os.path.join(self.root, path))
                    elif int(camera) == 7:
                        self.seven_files.append([])
                        if len(self.seven_files) < int(frame):
                            for _ in range(0, int(frame) - len(self.seven_files) + 1):
                                self.seven_files.append([])
                        # print("frame: %d, seven_files_len: %d" % (int(frame), len(self.seven_files)))
                        self.seven_files[int(frame)].append(os.path.join(self.root, path))
                        self.sum_files[int(frame)].append(os.path.join(self.root, path))
                    else:
                        self.files[int(frame)].append(os.path.join(self.root, path))
                        self.sum_files[int(frame)].append(os.path.join(self.root, path))
                else:
                    if int(camera) == focal_range[0]:
                        self.files.append([])
                        self.files[int(frame)].append(os.path.join(self.root, path))
                    elif int(camera) == 7:
                        self.seven_files.append([])
                        self.seven_files[int(frame)].append(os.path.join(self.root, path))
                    else:
                        self.files[int(frame)].append(os.path.join(self.root, path))

    # def __iter__(self):
    #     self.iter_idx += 1+
    #     return self.__getitem__(self.iter_idx) 차후 고려.. todo ㅇㅅㅇ

    def __getitem__(self, idx):
        self.iter_idx = idx
        # print('all_files',len(self.files[idx]))
        # print('seven_files', len(self.seven_files[idx]))

        # limit = len(self.files[idx]) if self.limit is None else self.limit - 1
        # print(self.files[idx][self.main_camera])
        # print('files_name', self.files[idx])
        # print('files_name', self.files)
        # print("frame_range: {}".format(self.frame_range[0]))
        # print("seven_files_70:", self.seven_files[70])
        # print(idx)
        idx += self.frame_range[0]
        images = [cv2.imread(self.files[idx][i], cv2.COLOR_BGR2RGB) for i in range(len(self.files[idx]))]
        seven_images = [cv2.imread(self.seven_files[idx][i], cv2.COLOR_BGR2RGB) for i in range(len(self.seven_files[idx]))]

        # seven_images = []
        # for i in range(5):
        #     print("asefasf:", self.seven_files[0][idx + self.frame_range[0]][i])
        focal_path = self.files[idx]
        # print('imagssss',len(images))
        # print('seven_imagssss', len(seven_images))
        # print("files: {}".format(self.files))
        # print("seven_files: {}".format(self.seven_files))
        # print("seven_images: {}".format(seven_images))
        # print("s")
        # [print(self.files[idx][i]) for i in range(len(self.files[idx]))]
        # cameras = [i for i in range(limit)]
        # images.insert(0, cv2.imread(self.files[idx][self.main_camera], cv2.COLOR_BGR2RGB))
        # cameras.insert(0, self.main_camera)
        return seven_images, images, focal_path, self.sum_files[idx]

    def __len__(self):
        return len(self.seven_files)


if __name__ == '__main__':
    dataloader = Load2DFolder(root='E:\\code\\SuperGluePretrainedNetwork-master\\unfold_images_color_1@', main_camera=7, frame_range=(0, 66), focal_range=None)
    print(np.array(dataloader.files).shape)
    print(np.array(dataloader.seven_files).shape)
    length = dataloader.__len__()
    print('length',length)

    img, focal, focal_path = dataloader[0]
    img1 = dataloader[66]
    a = len(img)
    a1 = len(focal)
    a2 = len(focal_path)
    b = len(img1)
    print(a)
    print(a1)
    print(a2)
    print(b)
    # print(img)


