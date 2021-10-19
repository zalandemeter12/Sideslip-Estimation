import numpy as np
import cv2
import math
import rosbag
from cv_bridge import CvBridge
from enum import Enum
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.util import img_as_float


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


def read_bounding_boxes_from_bag(bb_array):
    array = []
    for bounding_box in bb_array.detections:
        array.append([bounding_box.xmax, bounding_box.xmin, bounding_box.ymax, bounding_box.ymin, bounding_box.color])
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

    # contours = np.array([[0, 0], [0, 552], [420, 552], [875, 240], [1320, 240], [1730, 552], [2064, 552], [2064, 0]])
    # cv2.fillPoly(mask, pts=[contours], color=255)
    return mask


def find_bounding_box(bb, kp):
    for idx in range(len(bb)):
        if bb[idx][BB.xmax.value] >= kp[0] >= bb[idx][BB.xmin.value] and \
                bb[idx][BB.ymax.value] >= kp[1] >= bb[idx][BB.ymin.value]:
            return idx
    return None


def get_kp_ratio(kp, bb):
    x = kp[0] - bb[BB.xmin.value]
    y = kp[1] - bb[BB.ymin.value]
    w = bb[BB.xmax.value] - bb[BB.xmin.value]
    h = bb[BB.ymax.value] - bb[BB.ymin.value]
    if x == 0:
        x_ratio = 0
    else:
        x_ratio = w / x
    if y == 0:
        y_ratio = 0
    else:
        y_ratio = h / y
    return x_ratio, y_ratio


def is_good_match(query_kp, train_kp, query_properties, train_properties, pairs):
    query_idx = find_bounding_box(query_array, query_kp)
    train_idx = find_bounding_box(train_array, train_kp)

    if query_idx is None or train_idx is None:
        return False

    if query_properties[query_idx][PR.color.value] != train_properties[train_idx][PR.color.value]:
        return False

    if (query_idx, train_idx) not in pairs:
        return False

    query_x_ratio, query_y_ratio = get_kp_ratio(query_kp, query_array[query_idx])
    train_x_ratio, train_y_ratio = get_kp_ratio(train_kp, train_array[train_idx])

    X_LOWER_THRESHOLD = 0.9
    X_UPPER_THRESHOLD = 1.1
    Y_LOWER_THRESHOLD = 0.9
    Y_UPPER_THRESHOLD = 1.1

    if not (query_x_ratio * X_LOWER_THRESHOLD < train_x_ratio < query_x_ratio * X_UPPER_THRESHOLD) or \
            not (query_y_ratio * Y_LOWER_THRESHOLD < train_y_ratio < query_y_ratio * Y_UPPER_THRESHOLD):
        return False

    center_delta = (train_properties[train_idx][PR.cx.value] - query_properties[query_idx][PR.cx.value],
                    train_properties[train_idx][PR.cy.value] - query_properties[query_idx][PR.cy.value])

    delta = math.sqrt(center_delta[0] ** 2 + center_delta[1] ** 2)

    if delta > 75:
        return False

    return True


def draw_bounding_boxes(bb, img):
    for idx in range(len(bb)):
        if bb[idx][BB.color.value] == 0:
            cv2.rectangle(img, (bb[idx][BB.xmin.value], bb[idx][BB.ymin.value]),
                          (bb[idx][BB.xmax.value], bb[idx][BB.ymax.value]), (0, 255, 255), 2)
        elif bb[idx][BB.color.value] == 1:
            cv2.rectangle(img, (bb[idx][BB.xmin.value], bb[idx][BB.ymin.value]),
                          (bb[idx][BB.xmax.value], bb[idx][BB.ymax.value]), (255, 0, 0), 2)
        elif bb[idx][BB.color.value] == 3:
            cv2.rectangle(img, (bb[idx][BB.xmin.value], bb[idx][BB.ymin.value]),
                          (bb[idx][BB.xmax.value], bb[idx][BB.ymax.value]), (0, 140, 255), 2)


def post_process_images(bb_pairs, query_img, train_img, query_properties, train_properties, show_numbers):
    draw_bounding_boxes(query_array, query_img)
    draw_bounding_boxes(train_array, train_img)

    if show_numbers:

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


def draw_matches(query_keypoints, train_keypoints, good, mask, query_img, train_img, query_array, train_array):
    for bb in query_array:
        i = query_img[bb[BB.ymin.value]:bb[BB.ymax.value], bb[BB.xmin.value]:bb[BB.xmax.value]]
        img = np.array(i, dtype=float)
        a_channel = np.ones(img.shape, dtype=float) / 2.0
        image = img * a_channel
        train_img[bb[BB.ymin.value]:bb[BB.ymax.value], bb[BB.xmin.value]:bb[BB.xmax.value]] = (
                train_img[bb[BB.ymin.value]:bb[BB.ymax.value], bb[BB.xmin.value]:bb[BB.xmax.value]] + image)

    for i, tmp in enumerate(good):
        if mask[i] == 1:
            query_kp = query_keypoints[tmp[0].queryIdx].pt
            train_kp = train_keypoints[tmp[0].trainIdx].pt
            train_img = cv2.line(train_img, (int(query_kp[0]), int(query_kp[1])), (int(train_kp[0]), int(train_kp[1])),
                                 (0, 255, 0), 1)


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              pos=(0, 0),
              font_scale=1,
              font_thickness=1,
              text_color=(100, 200, 50),
              text_color_bg=(20, 20, 20)
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w + 10, y + text_h + 10), text_color_bg, -1)
    cv2.putText(img, text, (x + 5, y + text_h + font_scale - 1 + 5), font, font_scale, text_color, font_thickness)

    return text_size


###########################################################################


bag = rosbag.Bag('manual.bag')
bridge = CvBridge()
bb_list = []
img_list = []
for topic, msg, t in bag.read_messages(topics=['/perception_pylon_nodelet/image_rect', '/bounding_boxes']):
    if topic == '/perception_pylon_nodelet/image_rect':
        img_list.append((int(str(msg.header.stamp.secs) + str(msg.header.stamp.nsecs)), msg))
    elif topic == '/bounding_boxes':
        bb_list.append((int(str(msg.header.stamp.secs) + str(msg.header.stamp.nsecs)), msg))
bag.close()

detected_list = []
for img in img_list:
    for bb in bb_list:
        if img[0] == bb[0]:
            detected_list.append((img[1], bb[1]))

data_tx = []
data_ty = []
data_tz = []
data_rx = []
data_ry = []
data_rz = []
iters = []
sum_tz = 0
for i in range(len(detected_list) - 1):
    query_img = bridge.imgmsg_to_cv2(detected_list[i + 1][0], desired_encoding='passthrough').copy()
    train_img = bridge.imgmsg_to_cv2(detected_list[i][0], desired_encoding='passthrough').copy()

    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2BGRA)
    train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2BGRA)

    query_array = read_bounding_boxes_from_bag(detected_list[i + 1][1])
    train_array = read_bounding_boxes_from_bag(detected_list[i][1])

    query_properties = process_bounding_boxes(query_array, query_img)
    train_properties = process_bounding_boxes(train_array, train_img)

    query_dists = get_euclidean_dists(query_properties, query_properties)
    train_dists = get_euclidean_dists(train_properties, train_properties)

    # for count, value in enumerate(query_dists):
    #     for idx, dist in enumerate(value):
    #         if dist != 0.0 and dist < 10:
    #             if idx < len(query_array):
    #                 print(count, idx)
    #                 # query_array.pop(idx)
    #                 # query_properties.pop(idx)
    #                 # query_dists[idx][count] = 10000  # invalid large value
    #
    # for count, value in enumerate(train_dists):
    #     for idx, dist in enumerate(value):
    #         if dist != 0.0 and dist < 10:
    #             if idx < len(train_array):
    #                 print(count, idx)
    #                 # train_array.pop(idx)
    #                 # train_properties.pop(idx)
    #                 # train_dists[idx][count] = 10000  # invalid large value

    euclidean_dists = get_euclidean_dists(query_properties, train_properties)
    bb_pairs = pair_bbs(euclidean_dists)

    query_mask = create_mask(query_array)
    train_mask = create_mask(train_array)

    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=2)

    query_keypoints, queryDescriptors = orb.detectAndCompute(query_img, query_mask, None)
    train_keypoints, trainDescriptors = orb.detectAndCompute(train_img, train_mask, None)

    matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)

    matches = matcher.knnMatch(queryDescriptors=queryDescriptors, trainDescriptors=trainDescriptors, mask=None, k=2,
                               compactResult=False)

    good = []
    pts1 = []
    pts2 = []

    if len(matches) > 1:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # Ratio test
                query_kp = query_keypoints[m.queryIdx].pt
                train_kp = train_keypoints[m.trainIdx].pt

                if is_good_match(query_kp, train_kp, query_properties, train_properties, bb_pairs):
                    good.append([m])
                    pts1.append(query_kp)
                    pts2.append(train_kp)
    else:
        print("ValueError: not enough values to unpack (expected 2, got 1)")

    ski_query_keypoints = np.ndarray(shape=(len(query_keypoints), 2), dtype=float, order='F')
    for j, tmp in enumerate(query_keypoints):
        ski_query_keypoints[j][0] = tmp.pt[0]
        ski_query_keypoints[j][1] = tmp.pt[1]

    ski_train_keypoints = np.ndarray(shape=(len(train_keypoints), 2), dtype=float, order='F')
    for j, tmp in enumerate(train_keypoints):
        ski_train_keypoints[j][0] = tmp.pt[0]
        ski_train_keypoints[j][1] = tmp.pt[1]

    ski_matches = np.ndarray(shape=(len(good), 2), dtype=int, order='F')
    for j, tmp in enumerate(good):
        ski_matches[j][0] = tmp[0].queryIdx
        ski_matches[j][1] = tmp[0].trainIdx

    model, inliers = ransac((ski_query_keypoints[ski_matches[:, 0]],
                             ski_train_keypoints[ski_matches[:, 1]]),
                            FundamentalMatrixTransform, min_samples=8,
                            residual_threshold=1, max_trials=5000)

    mask = np.ndarray(shape=(len(inliers), 1), dtype=int, order='F')
    for j, tmp in enumerate(inliers):
        if tmp:
            mask[j] = 1
        else:
            mask[j] = 0

    print(mask)

    inlier_keypoints_left = ski_query_keypoints[ski_matches[inliers, 0]]
    inlier_keypoints_right = ski_train_keypoints[ski_matches[inliers, 1]]

    print(f'Number of matches: {ski_matches.shape[0]}')
    print(f'Number of inliers: {inliers.sum()}')

    # disp = inlier_keypoints_left[:, 1] - inlier_keypoints_right[:, 1]
    # disp_coords = np.round(inlier_keypoints_left).astype(np.int64)
    # disp_idxs = np.ravel_multi_index(disp_coords.T, groundtruth_disp.shape)
    # disp_error = np.abs(groundtruth_disp.ravel()[disp_idxs] - disp)
    # disp_error = disp_error[np.isfinite(disp_error)]

    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    plot_matches(ax[0], img_as_float(query_img), img_as_float(train_img), ski_query_keypoints, ski_train_keypoints,
                 ski_matches[inliers], only_matches=True)
    ax[0].axis("off")
    ax[0].set_title("Inlier correspondences")

    # ax[1].hist(disp_error)
    # ax[1].set_title("Histogram of disparity errors")

    plt.show()

    pts1 = np.float32(pts1).reshape(-1, 1, 2)
    pts2 = np.float32(pts2).reshape(-1, 1, 2)

    # K = np.array([[857.228760, 0.0, 1032.785009],
    #               [0.0, 977.735046, 38.772855],
    #               [0.0, 0.0, 1.0]])

    K = np.array([[1014.718468, 0.000000, 0],
                  [0.000000, 1018.165437, 0],
                  [1047.046240, -28.962809, 1.000000]])

    K[0, 0] = K[0, 0] * 0.9
    K[2, 1] = K[2, 1] * (-1)

    K = K.transpose()

    # print(i)
    # cnt = 0
    # while True:
    #
    #
    #
    #     print(t[2])
    #     print(cnt)
    #     cnt += 1
    #     if t[2] > 0.995:
    #         break

    try:
        True
    except:
        print("An exception occurred")

    print(len(mask))

    mask2 = mask.astype(np.uint8).copy()
    print(mask2)
    E, mask = cv2.findEssentialMat(points1=pts1, points2=pts2, cameraMatrix=K, method=cv2.RANSAC, prob=0.99,
                                   threshold=0.1, mask=mask2, maxIters=2000)

    _, R, t, mask = cv2.recoverPose(E=E, points1=pts1, points2=pts2, cameraMatrix=K, mask=mask2)

    print(len(mask))
    print(mask)

    R, _ = cv2.Rodrigues(R)

    iters.append(i)
    data_tx.append(t[2])
    data_ty.append(t[0])
    data_tz.append(t[1])
    data_rx.append(R[2])
    data_ry.append(R[0])
    data_rz.append(R[1])
    sum_tz += t[1]

    post_process_images(bb_pairs, query_img, train_img, query_properties, train_properties, True)

    # draw_matches(query_keypoints, train_keypoints, good, mask, query_img, train_img, query_array, train_array)

    final_img = cv2.drawMatchesKnn(query_img, query_keypoints, train_img, train_keypoints, good, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    blank_image = np.zeros((400, 185, 3), np.uint8)
    blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2BGRA)

    # final_img = cv2.hconcat([query_img, train_img])
    final_img = cv2.resize(final_img, (1665, 400))
    final_img = cv2.hconcat([blank_image, final_img])

    draw_text(final_img, "frame: " + str(i), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 10))
    draw_text(final_img, "matches: " + str(len(matches)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 40))
    draw_text(final_img, "good: " + str(len(good)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 70))
    draw_text(final_img, "mask: " + str(np.count_nonzero(mask)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 100))
    draw_text(final_img, "t[x]: " + str(round(t[2][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 130))
    draw_text(final_img, "t[y]: " + str(round(t[0][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 160))
    draw_text(final_img, "t[z]: " + str(round(t[1][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 190))
    draw_text(final_img, "R[x]: " + str(round(R[2][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 220))
    draw_text(final_img, "R[y]: " + str(round(R[0][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 250))
    draw_text(final_img, "R[z]: " + str(round(R[1][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 280))

    # final_img = cv2.vconcat([query_img, train_img])
    # final_img = cv2.resize(final_img, (1800, 800))

    # train_img = cv2.resize(train_img, (1665, 400))
    # train_img = cv2.hconcat([blank_image, train_img])
    # draw_text(train_img, "frame: " + str(i), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 10))
    # draw_text(train_img, "matches: " + str(len(matches)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 40))
    # draw_text(train_img, "good: " + str(len(good)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 70))
    # draw_text(train_img, "t[x]: " + str(round(t[2][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 100))
    # draw_text(train_img, "t[y]: " + str(round(t[0][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 130))
    # draw_text(train_img, "t[z]: " + str(round(t[1][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 160))
    # draw_text(train_img, "R[x]: " + str(round(R[2][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 190))
    # draw_text(train_img, "R[y]: " + str(round(R[0][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 220))
    # draw_text(train_img, "R[z]: " + str(round(R[1][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 250))

    cv2.imshow("Matches", final_img)
    cv2.waitKey()

plt.plot(iters, data_tx)
plt.xlabel('frames')
plt.ylabel('t[x]')
plt.title('Plot of the t[x] component')
plt.show()

plt.plot(iters, data_ty)
plt.xlabel('frames')
plt.ylabel('t[y]')
plt.title('Plot of the t[y] component')
plt.show()

plt.plot(iters, data_tz)
plt.xlabel('frames')
plt.ylabel('t[z]')
plt.title('Plot of the t[z] component')
plt.show()

plt.plot(iters, data_rx)
plt.xlabel('frames')
plt.ylabel('R[x]')
plt.title('Plot of the R[x] component')
plt.show()

plt.plot(iters, data_ry)
plt.xlabel('frames')
plt.ylabel('R[y]')
plt.title('Plot of the R[y] component')
plt.show()

plt.plot(iters, data_rz)
plt.xlabel('frames')
plt.ylabel('R[z]')
plt.title('Plot of the R[z] component')
plt.show()

print(sum_tz)