import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import pickle
import os
import a

MIN_MATCH_COUNT = 10


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


def keypoint_filter(images_path, images_pickle):
    images = []
    for image_path in images_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images.append(image)

    kp0, kp1, matches = orb_match(images[0], images[1])

    segmentation_results = []
    boxes0=[]
    boxes1=[]
    for i,image_pickle in enumerate(images_pickle):
        segmentation_result = load_pickle(image_pickle)[0][0]
        instance_number = len(segmentation_result['class_ids'])
        for idx in range(instance_number):
            box = segmentation_result['rois'][idx]
            if i == 0:
                boxes0.append(box)
            elif i == 1:
                boxes1.append(box)
            else:
                sys.exit(-1)

        print(segmentation_result['class_ids'])
        print(segmentation_result['masks'])
        print(segmentation_result['scores'])
        print(segmentation_result['rois'])

        segmentation_results.append(segmentation_results)

    m = None
    match_boxes = []
    for match in matches:
        pt0 = kp0[match.queryIdx].pt
        pt1 = kp1[match.trainIdx].pt
        m, bb0_idx, bb1_idx = a.count_match(pt0, pt1, boxes0, boxes1, m)
        match_boxes.append((bb0_idx, bb1_idx))
    print(m)

    # start filter
    max_count_m = np.argmax(m, 1)
    matches_filter = []
    for i, match in enumerate(matches):
        bb0_idx, bb1_idx = match_boxes[i]
        if bb0_idx is None or bb1_idx is None:
            pass
        elif bb1_idx != max_count_m[bb0_idx]:
            pass
        else:
            matches_filter.append(match)

    origin_match_img = cv2.drawMatches(images[0], kp0, images[1], kp1, matches, None, flags=2)
    filter_match_img = cv2.drawMatches(images[0], kp0, images[1], kp1, matches_filter, None, flags=2)
    return matches,matches_filter, origin_match_img, filter_match_img


if __name__ == '__main__':
    image0_pickle_dir = '/home/yzbx/image_0'
    image1_pickle_dir = '/home/yzbx/image_1'
    image0_dir = '/home/yzbx/07/image_0'
    image1_dir = '/home/yzbx/07/image_1'

    for filename in os.listdir(image0_dir):
        if not filename.endswith('png'):
            continue

        image0_path = os.path.join(image0_dir, filename)
        image1_path = os.path.join(image1_dir, filename)
        pkl0_path = os.path.join(image0_pickle_dir, filename.replace('png', 'pkl'))
        pkl1_path = os.path.join(image1_pickle_dir, filename.replace('png', 'pkl'))

        print(image0_path, image1_path, pkl0_path, pkl1_path)

        matches,filter_matches, origin_img, filter_img = keypoint_filter((image0_path, image1_path), (pkl0_path, pkl1_path))
        print('good match number', len(filter_matches))
        print('bad match number', len(matches) - len(filter_matches))
        # cv2.imshow('origin image match', origin_img)
        # cv2.imshow('filter image match', filter_img)

        # key = cv2.waitKey('30')
        # if key == 'q':
        #     break

        plt.subplot(121)
        plt.imshow(origin_img)
        plt.subplot(122)
        plt.imshow(filter_img)
        plt.show(block=True)
