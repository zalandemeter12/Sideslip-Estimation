import numpy as np
import cv2
import math
import rosbag
from enum import Enum


###########################################################################


class BB(Enum):
    xmax = 0
    xmin = 1
    ymax = 2
    ymin = 3
    color = 4


class PR(Enum):
    dist = 0
    angle = 1
    cx = 2
    cy = 3
    color = 4


def read_bounding_boxes(txt):
    array = []
    for line in txt.readlines():
        tmp = []
        for y in line.rstrip('\n').split(' '):
            tmp.append(int(y))
        array.append(tmp)
    txt.close()
    return array


def process_bounding_boxes(bb, img):
    properties = []
    for idx in range(len(bb)):
        center = (bb[idx][BB.xmin.value] + (bb[idx][BB.xmax.value] - bb[idx][BB.xmin.value]) * 0.5,
                  bb[idx][BB.ymin.value] + (bb[idx][BB.ymax.value] - bb[idx][BB.ymin.value]) * 0.5)
        color = bb[idx][BB.color.value]
        dist = math.sqrt(center[0] ** 2 + center[1] ** 2)
        if bb[idx][BB.xmax.value] == 0:
            angle = 0
        elif bb[idx][BB.ymax.value] == 0:
            angle = 90
        else:
            angle = math.asin(center[0] / dist) * 180 / math.pi
        properties.append((dist, angle, center[0], center[1], color))
        # cv2.line(img, (0, 0), (int(center[0]), int(center[1])), (255, 0, 0), 1)
        if bb[idx][BB.color.value] == 0:
            cv2.rectangle(img, (bb[idx][BB.xmin.value], bb[idx][BB.ymin.value]),
                          (bb[idx][BB.xmax.value], bb[idx][BB.ymax.value]), (0, 255, 255), 2)
        elif bb[idx][BB.color.value] == 1:
            cv2.rectangle(img, (bb[idx][BB.xmin.value], bb[idx][BB.ymin.value]),
                          (bb[idx][BB.xmax.value], bb[idx][BB.ymax.value]), (255, 0, 0), 2)
    return properties


def get_euclidean_dists(q_properties, t_properties):
    distances = []

    for qbb in q_properties:
        tmp_distances = []
        for tbb in t_properties:
            tmp_dist = math.sqrt(
                (qbb[PR.cx.value] - tbb[PR.cx.value]) ** 2 + (qbb[PR.cy.value] - tbb[PR.cy.value]) ** 2)
            if qbb[PR.color.value] == tbb[PR.color.value]:
                tmp_distances.append(tmp_dist)
            else:
                tmp_distances.append(10000)  # invalid large value
        distances.append(tmp_distances)

    return distances


def get_min_dists(dists):
    min_idxes = []
    min_dists = []
    for i in range(len(dists)):
        tmp_dist = min(dists[i])
        min_idxes.append(dists[i].index(tmp_dist))
        min_dists.append(tmp_dist)

    return min_idxes, min_dists


def pair_bbs(distances):
    pairs = []
    q_paired = [False] * len(distances)
    t_paired = [False] * len(distances[0])
    min_indexes, min_distances = get_min_dists(distances)

    shorter_idx = 0 if len(distances) < len(distances[0]) else 1
    shorter_len = len(distances) if shorter_idx == 0 else len(distances[0])
    true_num = 0
    while true_num < shorter_len:
        q_idx = min_distances.index(min(min_distances))
        t_idx = min_indexes[q_idx]
        pairs.append((q_idx, t_idx))
        q_paired[q_idx] = True
        t_paired[t_idx] = True
        true_num += 1
        for i in range(len(distances)):
            for j in range(len(distances[i])):
                if i == q_idx or j == t_idx:
                    distances[i][j] = 10000  # invalid large value
        min_indexes, min_distances = get_min_dists(distances)

    if shorter_idx == 0:
        for t_fidx in range(len(t_paired)):
            if not t_paired[t_fidx]:
                pairs.append((None, t_fidx))
    else:
        for q_fidx in range(len(q_paired)):
            if not q_paired[q_fidx]:
                pairs.append((q_fidx, None))

    return pairs


def create_mask(array):
    mask = np.zeros(query_img.shape[:2], dtype=np.uint8)
    for bb in array:
        cv2.rectangle(mask, (bb[BB.xmax.value], bb[BB.ymax.value]),
                      (bb[BB.xmin.value], bb[BB.ymin.value]), 255, thickness=-1)
    return mask


def find_bounding_box(bb, kp):
    for idx in range(len(bb)):
        if bb[idx][BB.xmax.value] > kp[0] > bb[idx][BB.xmin.value] and \
                bb[idx][BB.ymax.value] > kp[1] > bb[idx][BB.ymin.value]:
            return idx
    return None


def is_good_match(p_query_idx, p_train_idx, p_query_properties, p_train_properties, pairs):
    if p_query_idx is None or p_train_idx is None:
        return False

    if p_query_properties[p_query_idx][PR.color.value] != p_train_properties[p_train_idx][PR.color.value]:
        return False

    if (p_query_idx, p_train_idx) not in pairs:
        return False

    center_delta = (p_train_properties[p_train_idx][PR.cx.value] - p_query_properties[p_query_idx][PR.cx.value],
                    p_train_properties[p_train_idx][PR.cy.value] - p_query_properties[p_query_idx][PR.cy.value])

    delta = math.sqrt(center_delta[0] ** 2 + center_delta[1] ** 2)

    if delta > 100:
        return False

    return True


###########################################################################


first = '1955'
second = '1954'

query_img = cv2.imread(first + '.bmp')  # first
train_img = cv2.imread(second + '.bmp')  # second

query_txt = open(first + '.txt', 'r')
train_txt = open(second + '.txt', 'r')

query_array = read_bounding_boxes(query_txt)
train_array = read_bounding_boxes(train_txt)

timer = cv2.getTickCount()

query_properties = process_bounding_boxes(query_array, query_img)
train_properties = process_bounding_boxes(train_array, train_img)

euclidean_dists = get_euclidean_dists(query_properties, train_properties)
bb_pairs = pair_bbs(euclidean_dists)

for idx in range(len(bb_pairs)):
    q_idx = bb_pairs[idx][0]
    t_idx = bb_pairs[idx][1]
    if q_idx is not None and t_idx is not None:
        cv2.putText(query_img, str(q_idx),
                    (int(query_properties[q_idx][PR.cx.value]), int(query_properties[q_idx][PR.cy.value])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(train_img, str(q_idx),
                    (int(train_properties[t_idx][PR.cx.value]), int(train_properties[t_idx][PR.cy.value])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    elif q_idx is not None and t_idx is None:
        cv2.putText(query_img, str(q_idx),
                    (int(query_properties[q_idx][PR.cx.value]), int(query_properties[q_idx][PR.cy.value])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    elif q_idx is None and t_idx is not None:
        cv2.putText(train_img, str(t_idx),
                    (int(train_properties[t_idx][PR.cx.value]), int(train_properties[t_idx][PR.cy.value])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

query_mask = create_mask(query_array)
train_mask = create_mask(train_array)

orb = cv2.ORB_create(
    nfeatures=1000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=2)

query_keypoints, queryDescriptors = orb.detectAndCompute(query_img, query_mask, None)
train_keypoints, trainDescriptors = orb.detectAndCompute(train_img, train_mask, None)

matcher = cv2.BFMatcher(
    normType=cv2.NORM_L2,
    crossCheck=False
)

matches = matcher.knnMatch(
    queryDescriptors=queryDescriptors,
    trainDescriptors=trainDescriptors,
    mask=None,
    k=2,
    compactResult=False
)

# Apply ratio test
good = []
pts1 = []
pts2 = []
print(len(matches))
if len(matches) > 1:
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Ratio test
            query_kp = query_keypoints[m.queryIdx].pt
            train_kp = train_keypoints[m.trainIdx].pt

            query_idx = find_bounding_box(query_array, query_kp)
            train_idx = find_bounding_box(train_array, train_kp)

            if is_good_match(query_idx, train_idx, query_properties, train_properties, bb_pairs):
                good.append([m])
                pts1.append(query_kp)
                pts2.append(train_kp)
else:
    print("ValueError: not enough values to unpack (expected 2, got 1)")
print(len(good))

pts1 = np.float32(pts1).reshape(-1, 1, 2)
pts2 = np.float32(pts2).reshape(-1, 1, 2)

K = np.array([[857.228760, 0.0, 1032.785009],
              [0.0, 977.735046, 38.772855],
              [0.0, 0.0, 1.0]])
# D = np.array([[-0.018682808343432777], [-0.044315351694893736], [0.047678551616171246], [-0.018283908577445218]])
#
# # Remove the fisheye distortion from the points
# pts0 = cv2.fisheye.undistortPoints(pts0, K, D, P=K)
# pts2 = cv2.fisheye.undistortPoints(pts2, K, D, P=K)

E, mask = cv2.findEssentialMat(
    points1=pts1,
    points2=pts2,
    cameraMatrix=K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0,
    mask=None,
    maxIters=1000
)

_, R, t, mask = cv2.recoverPose(
    E=E,
    points1=pts1,
    points2=pts2,
    cameraMatrix=K,
    mask=mask)

R, _ = cv2.Rodrigues(R)

print(R)
print(t)
print("rx = ", R[2])
print("ry = ", R[0])
print("rz = ", R[1])
print("tx = ", t[2])
print("ty = ", t[0])
print("tz = ", t[1])

final_img = cv2.drawMatchesKnn(query_img, query_keypoints, train_img, train_keypoints, good, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

final_img = cv2.resize(final_img, (1800, 400))

# final_img = cv2.vconcat([query_img, train_img])
# final_img = cv2.resize(final_img, (1800, 800))

cv2.imshow("Matches", final_img)
cv2.waitKey()
