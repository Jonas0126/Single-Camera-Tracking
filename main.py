import torch
import os
import cv2
import numpy as np
from argparse import ArgumentParser
from FrameLoader import FrameLoader
from Cropper import Cropper
from Matcher import Matcher
from torchvision import transforms
from tqdm import tqdm
from model import Resnet101IbnA
class Palette:
    def __init__(self):     
        self.colors = {}
        
    def get_color(self, id):
        if not id in self.colors:
            color = list(np.random.choice(range(256), size=3))
            color = (int(color[0]), int(color[1]), int(color[2]))

            self.colors[id] = color

        return self.colors[id]





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--frame_dir', '-f', type=str)
    parser.add_argument('--label_dir', '-l', type=str)
    parser.add_argument('--detail', type=str, default='off', help='record all information include dist_matrix etc.')
    parser.add_argument('--cam', '-c', type=int, help='the camera you want to visualize')
    parser.add_argument('--params', '-p', type=str, help='the path to the pytorch model')
    parser.add_argument('--out', type=str, help='the path to save the video')
    parser.add_argument('--name', type=str)
    parser.add_argument('--width', '-w', type=int)
    args = parser.parse_args()

    if not os.path.exists(f'{args.out}'):
        os.mkdir(args.out)

    # load data
    frameloader = FrameLoader(args.frame_dir, args.label_dir)
    imgs, labels = frameloader.load(args.cam)
    

    save_path = os.path.join(f'{args.out}', f'video/{args.name}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    out = cv2.VideoWriter(save_path, fourcc, 2, (1280,  720)) 


    #load feature extractor
    extracter = Resnet101IbnA()
    extracter = torch.load(args.params)
    extracter = extracter.to('cpu')
    extracter.eval()

    #normalize img
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cropper = Cropper(args.width)
    matcher = Matcher(threshold=0.5, buffer_size=1)
    palette = Palette()
    with torch.no_grad():
        for i in tqdm(range(len(imgs))):
            current_objects = []    
            object_embeddings = []
            info_list = []

            current_objects, info_list = cropper.crop_frame(image_path=imgs[i], label_path=labels[i])

            #get embeddings
            for j in range(len(current_objects)):
                img = transform(current_objects[j])
                
                _, feature, _ = extracter(torch.unsqueeze(img,0))
                object_embeddings.append(torch.squeeze(feature))

            id_list =  matcher.match(object_embeddings, info_list)

            #draw bounding box
            image = cv2.imread(imgs[i])
            for n in range(len(info_list)):

                color = palette.get_color(id_list[n])
                cv2.rectangle(image, (info_list[n][0], info_list[n][1]), (info_list[n][2], info_list[n][3]), color, 2)
                cv2.putText(image, text=str(id_list[n]), org=(info_list[n][0], info_list[n][1] - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color, thickness=3)

            out.write(image)
    out.release()


