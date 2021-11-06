import numpy as np
import cv2
import math
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


def read_bounding_boxes_from_bag(bb_array):
    array = []
    for bounding_box in bb_array.detections:
        array.append([bounding_box.xmax, bounding_box.xmin, bounding_box.ymax, bounding_box.ymin, bounding_box.color])
    return array


def process_bounding_boxes(bb):
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
    mask = np.zeros((552, 2064), dtype=np.uint8)
    for bb in array:
        cv2.rectangle(mask, (bb[BB.xmax.value], bb[BB.ymax.value]),
                      (bb[BB.xmin.value], bb[BB.ymin.value]), 255, thickness=-1)

    # contours = np.array([[0, 0], [0, 552], [420, 552], [875, 240], [1320, 240], [1730, 552], [2064, 552], [2064, 0]])
    # cv2.fillPoly(mask, pts=[contours], color=255)
    return mask


def cvkp_to_skikp(keypoints):
    ski_keypoints = np.ndarray(shape=(len(keypoints), 2), dtype=float, order='F')
    for j, tmp in enumerate(keypoints):
        ski_keypoints[j][0] = tmp.pt[0]
        ski_keypoints[j][1] = tmp.pt[1]
    return ski_keypoints


def cvmatch_to_skimatch(matches):
    ski_matches = np.ndarray(shape=(len(matches), 2), dtype=int, order='F')
    for j, tmp in enumerate(matches):
        ski_matches[j][0] = tmp[0].queryIdx
        ski_matches[j][1] = tmp[0].trainIdx
    return ski_matches


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


def is_good_match(query_kp, train_kp, query_properties, train_properties, pairs, query_array, train_array):
    # query_idx = find_bounding_box(query_array, query_kp)
    # train_idx = find_bounding_box(train_array, train_kp)
    #
    # if query_idx is None or train_idx is None:
    #     return False
    #
    # if query_properties[query_idx][PR.color.value] != train_properties[train_idx][PR.color.value]:
    #     return False
    #
    # # if (query_idx, train_idx) not in pairs:
    # #     return False
    #
    # query_x_ratio, query_y_ratio = get_kp_ratio(query_kp, query_array[query_idx])
    # train_x_ratio, train_y_ratio = get_kp_ratio(train_kp, train_array[train_idx])
    #
    # X_LOWER_THRESHOLD = 0.9
    # X_UPPER_THRESHOLD = 1.1
    # Y_LOWER_THRESHOLD = 0.9
    # Y_UPPER_THRESHOLD = 1.1
    #
    # if not (query_x_ratio * X_LOWER_THRESHOLD < train_x_ratio < query_x_ratio * X_UPPER_THRESHOLD) or \
    #         not (query_y_ratio * Y_LOWER_THRESHOLD < train_y_ratio < query_y_ratio * Y_UPPER_THRESHOLD):
    #     return False
    #
    # # center_delta = (train_properties[train_idx][PR.cx.value] - query_properties[query_idx][PR.cx.value],
    # #                 train_properties[train_idx][PR.cy.value] - query_properties[query_idx][PR.cy.value])
    # #
    # # delta = math.sqrt(center_delta[0] ** 2 + center_delta[1] ** 2)
    # #
    # # if delta > 75:
    # #     return False

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


def post_process_images(bb_pairs, query_img, train_img, query_properties, train_properties, show_numbers, query_array, train_array):
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


def make_final_image(query_img, query_keypoints, train_img, train_keypoints, good, mask, i, matches, t, R):
    draw_params = dict(matchesMask=mask,
                       matchColor=(82, 168, 50),
                       flags=2)

    final_img = cv2.drawMatchesKnn(query_img, query_keypoints, train_img, train_keypoints, good, None,
                                   **draw_params)

    blank_image = np.zeros((400, 185, 3), np.uint8)

    # final_img = cv2.hconcat([query_img, train_img])
    final_img = cv2.resize(final_img, (1665, 400))
    final_img = cv2.hconcat([blank_image, final_img])

    draw_text(final_img, "time: " + str(i/20) , font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 10))
    draw_text(final_img, "matches: " + str(len(matches)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 40))
    draw_text(final_img, "good: " + str(len(good)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 70))
    draw_text(final_img, "inliers: " + str(np.count_nonzero(mask)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 100))
    draw_text(final_img, "t[x]: " + str(round(t[2][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 130))
    draw_text(final_img, "t[y]: " + str(round(t[0][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 160))
    draw_text(final_img, "t[z]: " + str(round(t[1][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 190))
    draw_text(final_img, "R[x]: " + str(round(R[2][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 220))
    draw_text(final_img, "R[y]: " + str(round(R[0][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 250))
    draw_text(final_img, "R[z]: " + str(round(R[1][0], 8)), font=cv2.FONT_HERSHEY_PLAIN, pos=(10, 280))

    return final_img

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