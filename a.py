import numpy as np
import cv2
import matplotlib.pyplot as plt


def count_match(pt1, pt2, boxes1, boxes2, m):
    if m is None:
        m = np.zeros((len(boxes1), len(boxes2)), np.int)
    assert m.shape == (len(boxes1), len(boxes2))

    x, y = pt1
    pt1_in_boxes1_idx = None
    for idx, box1 in enumerate(boxes1):
        x1, y1, x2, y2 = box1
        if x1 < x < x2 and y1 < y < y2:
            pt1_in_boxes1_idx = idx
            break

    x, y = pt2
    pt2_in_boxes2_idx = None
    for idx, box2 in enumerate(boxes2):
        x1, y1, x2, y2 = box2
        if x1 < x < x2 and y1 < y < y2:
            pt2_in_boxes2_idx = idx
            break

    if pt1_in_boxes1_idx is not None and pt2_in_boxes2_idx is not None:
        m[pt1_in_boxes1_idx, pt2_in_boxes2_idx] += 1

    return m,pt1_in_boxes1_idx,pt2_in_boxes2_idx

if __name__ == '__main__':

    img1 = cv2.imread('/home/yzbx/1.jpg',0) # queryImage
    img2 = cv2.imread('/home/yzbx/2.jpg',0) # trainImage
    # Initiate ORB detector
    orb = cv2.ORB_create(1000)
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    print(kp1)
    for kp in kp1:
        # print(kp.pt)
        x=kp.pt[0]
        y=kp.pt[1]

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:1000],None, flags=2)

    boxes1=None
    boxes2=None
    m=None
    for match in matches:
        pt1=kp1[match.queryIdx].pt
        pt2=kp2[match.trainIdx].pt
        m = count_match(pt1,pt2,boxes1,boxes2,m)

    plt.imshow(img3),plt.show()