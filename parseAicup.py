from argparse import ArgumentParser
from collections import defaultdict
import os


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--save_dir', '-s', type=str, help='Directory containing input video frames.')
    parser.add_argument('--label_dir', '-l', type=str, help='Directory containing labels for input frames.')
    parser.add_argument('-date', '-d', type=str)
    parser.add_argument('-cam', '-c', type=int)
    args = parser.parse_args()

    
    camera_labels = defaultdict(list)
    for file in os.listdir(args.label_dir):
        camera_id = int(file[0])
        camera_labels[camera_id].append(os.path.join(args.label_dir, file))
    for k, v in camera_labels.items():
        v.sort()

    
    labels = camera_labels[args.cam]
    f = open(f'{args.save_dir}/{args.date}_{args.cam}.txt', 'w')
    for i in range(len(labels)):
        label = open(labels[i], 'r')
        info = label.readline()
        while info:
            f.write(f'{info}')
            info = label.readline()
