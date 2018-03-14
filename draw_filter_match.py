import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import pickle
import os
import a
import visualize

MIN_MATCH_COUNT = 10
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def load_pickle(filename):
    with open(filename, 'rb') as save_f:
        results = pickle.load(save_f)
        return results


def orb_match(img1, img2):
    orb = cv2.ORB_create(1000)
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # print(kp1)
    # for kp in kp1:
    #     # print(kp.pt)
    #     x = kp.pt[0]
    #     y = kp.pt[1]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches

def draw_mask_image(image_path,result):
    masks=result['masks']
    height,width,N=masks.shape
    colors =visualize.random_colors(N)

    image=cv2.imread(image_path,cv2.IMREAD_COLOR)

    masked_image=image.astype(np.uint32).copy()
    for i in range(N):
        mask = masks[:, :, i]
        masked_image = visualize.apply_mask(masked_image, mask, colors[i],alpha=1)

    return masked_image.astype(np.uint8)

def check_match_people(pt0,pt1,masks0,masks1):
    x, y = pt0
    pt1_in_boxes1_idx = []
    height,width,channel=masks0.shape

    for idx in range(channel):
        if masks0[int(y),int(x),idx]==1:
            pt1_in_boxes1_idx.append(idx)

    x, y = pt1
    pt2_in_boxes2_idx = []
    height, width, channel = masks1.shape
    for idx in range(channel):
        if masks1[int(y),int(x),idx]==1:
            pt2_in_boxes2_idx.append(idx)
    
    return False
def keypoint_mask_filter(images_path, images_pickle):
    images = []
    for image_path in images_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print('bad image path',image_path)
        images.append(image)

    print('image[0] shape is',images[0].shape)
    print('image[1] shape is', images[1].shape)
    kp0, kp1, matches = orb_match(images[0], images[1])

    masks = []

    idx=0
    mask_images=[]
    for image_pickle in images_pickle:
        r = load_pickle(image_pickle)[0][0]
        masks.append(r['masks'])
        mask_image=draw_mask_image(images_path[idx],r)
        mask_images.append(mask_image)
        idx+=1
    
    matches_filter=[]
    for match in matches:
        pt0 = kp0[match.queryIdx].pt
        pt1 = kp1[match.trainIdx].pt
        match_people = check_match_people(pt0, pt1, masks[0], masks[1])
        if match_people:
            matches_filter.append(match)

            
    print('len of matches',len(matches))
    print('len of filtered matches',len(matches_filter))
    
    origin_match_img = cv2.drawMatches(mask_images[0], kp0, mask_images[1], kp1, matches, None, flags=0)
    filter_match_img = cv2.drawMatches(mask_images[0], kp0, mask_images[1], kp1, matches, None, flags=0)
    return matches, matches_filter, origin_match_img, filter_match_img


if __name__ == '__main__':
    image0_pickle_dir = '/home/yzbx/git/Match-MaskRCNN/data/pkl/image_0'
    image1_pickle_dir = '/home/yzbx/git/Match-MaskRCNN/data/pkl/image_1'
    image0_dir = '/home/yzbx/git/Match-MaskRCNN/data/image/image_0'
    image1_dir = '/home/yzbx/git/Match-MaskRCNN/data/image/image_1'

    image_format='jpg'
    for filename0 in os.listdir(image0_dir):
        if not filename0.endswith(image_format):
            continue
        image0_path = os.path.join(image0_dir, filename0)
        pkl0_path = os.path.join(image0_pickle_dir, filename0.replace(image_format, 'pkl'))

        filename1=filename0.replace('L','R')
        image1_path = os.path.join(image1_dir, filename1)
        pkl1_path = os.path.join(image1_pickle_dir, filename1.replace(image_format, 'pkl'))

        print(image0_path, image1_path, pkl0_path, pkl1_path)
        assert os.path.exists(image0_path)
        assert os.path.exists(image1_path)
        assert os.path.exists(pkl0_path)
        assert os.path.exists(pkl1_path)

        # matches,filter_matches, origin_img, filter_img = keypoint_bbox_filter((image0_path, image1_path),
        #                                                                       (pkl0_path, pkl1_path))
        matches, filter_matches, origin_img, filter_img = keypoint_mask_filter((image0_path, image1_path),
                                                                               (pkl0_path, pkl1_path))
        print('good match number', len(filter_matches))
        print('bad match number', len(matches) - len(filter_matches))
        # cv2.imshow('origin image match', origin_img)
        # cv2.imshow('filter image match', filter_img)

        # key = cv2.waitKey('30')
        # if key == 'q':
        #     break

        plt.subplot(211)
        plt.imshow(origin_img)
        plt.subplot(212)
        plt.imshow(filter_img)
        plt.show(block=True)
