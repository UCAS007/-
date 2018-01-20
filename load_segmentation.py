import pickle
import cv2
import os
import numpy as np

ROOT_Image_DIR='/home/yzbx/Pictures/07'
ROOT_Segmentation_DIR='/home/yzbx/Pictures/output'

def load_pickle(filename):
    with open(filename, 'rb') as save_f:
        results = pickle.load(save_f)
        return results

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

# y1, x1, y2, x2 = boxes[i]
# mask = masks[:,:,i]
for image_0_file in os.listdir(os.path.join(ROOT_Image_DIR,'image_0')):
    image_0=cv2.imread(os.path.join(ROOT_Image_DIR,'image_0',image_0_file))
    image_1=cv2.imread(os.path.join(ROOT_Image_DIR,'image_1',image_0_file))
    segmentation_result_0=load_pickle(os.path.join(ROOT_Segmentation_DIR,
                                                   'image_0',
                                                   image_0_file.replace('png','pkl')))[0][0]
    segmentation_result_1 = load_pickle(os.path.join(ROOT_Segmentation_DIR,
                                                     'image_1',
                                                     image_0_file.replace('png', 'pkl')))[0][0]

    instance_number=len(segmentation_result_0['class_ids'])
    for idx in range(instance_number):
        box_0=segmentation_result_0['rois'][idx]

    print(segmentation_result_0['class_ids'])
    print(segmentation_result_0['masks'])
    print(segmentation_result_0['scores'])
    print(segmentation_result_0['rois'])
    # print(segmentation_result_1)

    break

