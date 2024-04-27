import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from torchvision.io import read_image

class Cropper():

    def __init__(self, img_width):
        self.img_width = img_width

    def convert(self, W, H, x_center_norm, y_center_norm, w_norm, h_norm):
        x_center = x_center_norm * W
        y_center = y_center_norm * H
        w = int(w_norm * W)
        h = int(h_norm * H)
        left = int(x_center - (w/2))
        top =  int(y_center - (h/2))
        
        return left, top, w, h

    def get_image_info(self, info, W, H):
        x_center_norm = float(info[1])
        y_center_norm = float(info[2])
        w_norm = float(info[3])
        h_norm = float(info[4])

        left, top, w, h = self.convert(W, H, x_center_norm, y_center_norm, w_norm, h_norm)
        
        return left, top, w, h


    def crop_frame(self, image_path, label_path):
        image = read_image(image_path)
        H = image.shape[1]
        W = image.shape[2]

        #get bounding box info
        label = open(label_path, 'r')
        info_list = []
        info = label.readline()

        cropped_regions = torch.empty((0, 3, self.img_width, self.img_width))
        
        while info:
            info = info.split(' ')
            left, top, w, h = self.get_image_info(info, W, H)
            info_list.append([left, top, left+w, top+h])
            #crop img
            croped_img = (F.crop(image, top, left, h, w))/255
            transform = transforms.Compose([
                transforms.Resize((self.img_width, self.img_width))
            ])
            croped_img = transform(croped_img)
            cropped_regions = torch.cat((cropped_regions, torch.unsqueeze(croped_img, 0)))
            info = label.readline()
        
        return cropped_regions, info_list