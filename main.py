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

# Define a class to manage color palettes for different IDs
class Palette:
    def __init__(self):     
        self.colors = {}
        
    def get_color(self, id):
        # Generate a random color for a new ID or return an existing color for a known ID
        if not id in self.colors:
            color = list(np.random.choice(range(256), size=3))
            color = (int(color[0]), int(color[1]), int(color[2]))

            self.colors[id] = color

        return self.colors[id]





if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--frame_dir', '-f', type=str, help='Directory containing input video frames.')
    parser.add_argument('--label_dir', '-l', type=str, help='Directory containing labels for input frames.')
    parser.add_argument('--model', '-m', type=str, default='resnet101_ibn_a', help='the name of the pre-trained PyTorch model')
    parser.add_argument('--out', type=str, help='Directory to save the output video.')
    parser.add_argument('--width', '-w', type=int, default=256)
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--visualize', '-v', type=str, default=False, help='Set to "True" to enable visualization of tracking results.')
    
    args = parser.parse_args()


    # Create output directory if it does not exist
    if not os.path.exists(f'{args.out}'):
        os.mkdir(args.out)

    # Set up the FrameLoader to load frames
    frameloader = FrameLoader(args.frame_dir, args.label_dir)

    # Load the pre-trained model for feature extraction
    extracter = torch.hub.load('b06b01073/veri776-pretrain', args.model, fine_tuned=True) # 將 fine_tuned 設為 True 會 load fine-tuned 後的 model
    extracter = extracter.to('cpu')
    extracter.eval()

    # Normalize image pixels before feeding them to the model
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    
    
    # Iterate over each camera
    for cam in range(8):
        
        # Initialize frame ID for writing to output file
        frame_id = 1

        # Load data for the current camera
        imgs, labels = frameloader.load(cam)

        # Create video writer if visualization is enabled
        if args.visualize:
            save_path = os.path.join(f'{args.out}', f'video/{args.name}_cam{cam}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(save_path, fourcc, 2, (1280,  720)) 

        # Initialize Cropper and Matcher
        cropper = Cropper(args.width)
        matcher = Matcher(threshold=0.5, buffer_size=1)
        palette = Palette()

        # Perform object tracking for each frame
        with torch.no_grad():
            for i in tqdm(range(len(imgs))):
                current_objects = []    
                object_embeddings = []
                info_list = []
                info_list_norm = []
                
                # Open a text file to record the label of each frame
                f = open(f'{args.out}/{cam}_{frame_id:05}.txt', 'w')

                # Crop objects from the current frame
                current_objects, info_list, info_list_norm = cropper.crop_frame(image_path=imgs[i], label_path=labels[i])

                # Extract features for each cropped object
                for j in range(len(current_objects)):
                    img = transform(current_objects[j])
                    
                    _, feature, _ = extracter(torch.unsqueeze(img,0))
                    object_embeddings.append(torch.squeeze(feature))

                # Match object embeddings to previous frames
                id_list =  matcher.match(object_embeddings, info_list)

                # Record coordinates and IDs to the output file
                for n in range(len(info_list)):
                    f.write(f'{cam} {info_list_norm[n][0]} {info_list_norm[n][1]} {info_list_norm[n][2]} {info_list_norm[n][3]} {id_list[n]}\n')
                frame_id += 1

                # Draw bounding boxes if visualization is enabled
                if args.visualize:
                    image = cv2.imread(imgs[i])
                    for n in range(len(info_list)):
                        color = palette.get_color(id_list[n])
                        cv2.rectangle(image, (info_list[n][0], info_list[n][1]), (info_list[n][2], info_list[n][3]), color, 2)
                        cv2.putText(image, text=str(id_list[n]), org=(info_list[n][0], info_list[n][1] - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color, thickness=3)

                    out.write(image)

        # Release video writer if visualization is enabled
        if args.visualize:
            out.release()


