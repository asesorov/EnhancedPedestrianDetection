import os
from os import listdir
from os.path import isfile, join
import argparse
import albumentations as A
import cv2


def apply_augmentations(input_dir, augmentations):
    files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
        
    for f in files:
        f = input_dir + '/' + f
        image = cv2.imread(f)
        augmented_image = A.Compose(augmentations)(image=image)['image']
        cv2.imwrite(f, augmented_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', type=str, required=True)
    args = parser.parse_args()

    transform = A.Compose([
        A.CLAHE(always_apply=True, p=1.0, clip_limit=1.6260416991257398, tile_grid_size=(2, 2)),
        A.Equalize(always_apply=True, p=1.0),
        A.ColorJitter(always_apply=True, p=1.0, brightness=0.18760935947326096,
                      contrast=0.2968125662449388, saturation=0, hue=0),
    ])

    apply_augmentations(args.input_dir, transform)
