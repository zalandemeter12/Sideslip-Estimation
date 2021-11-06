import rosbag
from cv_bridge import CvBridge
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from datetime import timedelta
import time
from utils import *
from plotting import *

# bag = rosbag.Bag('manual.bag')
# bridge = CvBridge()
# bb_list = []
# img_list = []
# imu_list = []
# imu_packets = []
# c = 0
# for topic, msg, t in bag.read_messages(
#         topics=['/perception_pylon_nodelet/image_rect', '/bounding_boxes', '/vectornav/IMU']):
#     if topic == '/perception_pylon_nodelet/image_rect':
#         img_list.append((int(str(msg.header.stamp.secs) + str(msg.header.stamp.nsecs)), msg))
#     elif topic == '/bounding_boxes':
#         bb_list.append((int(str(msg.header.stamp.secs) + str(msg.header.stamp.nsecs)), msg))
#     elif topic == '/vectornav/IMU':
#         imu_list.append(msg)
#         imu_packets.append(c)
#         c += 1
# bag.close()
#
# imu_rx = []
# imu_ry = []
# imu_rz = []
# for tmp in imu_list:
#     imu_rx.append(tmp.angular_velocity.x)
#     imu_ry.append(tmp.angular_velocity.y)
#     imu_rz.append(tmp.angular_velocity.z)
#
#
# detected_list = []
# for img in img_list:
#     for bb in bb_list:
#         if img[0] == bb[0]:
#             detected_list.append((img[1], bb[1]))
#
# data_tx, data_ty, data_tz = [], [], []
# data_rx, data_ry, data_rz = [], [], []
# data_matches, data_good, data_inliers = [], [], []
# data_sumtz, data_iters, data_sideslip = [], [], []
# data_frames = []
#
# sum_tz = 0
# start = time.time()
# for i in range(600, len(detected_list) - 1):
#     query_img = bridge.imgmsg_to_cv2(detected_list[i + 1][0], desired_encoding='passthrough').copy()
#     train_img = bridge.imgmsg_to_cv2(detected_list[i][0], desired_encoding='passthrough').copy()
#
#     query_array = read_bounding_boxes_from_bag(detected_list[i + 1][1])
#     train_array = read_bounding_boxes_from_bag(detected_list[i][1])
#
#     query_properties = process_bounding_boxes(query_array)
#     train_properties = process_bounding_boxes(train_array)
#
#     # query_dists = get_euclidean_dists(query_properties, query_properties)
#     # train_dists = get_euclidean_dists(train_properties, train_properties)
#     #
#     # for count, value in enumerate(query_dists):
#     #     for idx, dist in enumerate(value):
#     #         if dist != 0.0 and dist < 10:
#     #             if idx < len(query_array):
#     #                 print(count, idx)
#     #                 query_array.pop(idx)
#     #                 query_properties.pop(idx)
#     #                 query_dists[idx][count] = 10000  # invalid large value
#     #
#     # for count, value in enumerate(train_dists):
#     #     for idx, dist in enumerate(value):
#     #         if dist != 0.0 and dist < 10:
#     #             if idx < len(train_array):
#     #                 print(count, idx)
#     #                 train_array.pop(idx)
#     #                 train_properties.pop(idx)
#     #                 train_dists[idx][count] = 10000  # invalid large value
#
#     euclidean_dists = get_euclidean_dists(query_properties, train_properties)
#     bb_pairs = pair_bbs(euclidean_dists)
#
#     query_mask = create_mask(query_array)
#     train_mask = create_mask(train_array)
#
#     orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2,
#                          scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=2)
#
#     query_keypoints, queryDescriptors = orb.detectAndCompute(query_img, query_mask, None)
#     train_keypoints, trainDescriptors = orb.detectAndCompute(train_img, train_mask, None)
#
#     matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False) #NORM_HAMMING
#
#     matches = matcher.knnMatch(queryDescriptors=queryDescriptors, trainDescriptors=trainDescriptors, mask=None, k=2,
#                                compactResult=False)
#
#     # # CUDA
#     # cuMat1 = cv2.cuda_GpuMat()
#     # cuMat2 = cv2.cuda_GpuMat()
#     # cuMat1.upload(query_img)
#     # cuMat2.upload(train_img)
#     # cuMat1g = cv2.cuda.cvtColor(cuMat1, cv2.COLOR_RGB2GRAY)
#     # cuMat2g = cv2.cuda.cvtColor(cuMat2, cv2.COLOR_RGB2GRAY)
#     #
#     # cuQueryMask = cv2.cuda_GpuMat()
#     # cuTrainMask = cv2.cuda_GpuMat()
#     # cuQueryMask.upload(query_mask)
#     # cuTrainMask.upload(train_mask)
#
#     # # ORB
#     # corb = cv2.cuda_ORB.create(nfeatures=5000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2,
#     #                            scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=2, blurForDescriptor=False)
#     #
#     # _kps1, _descs1 = corb.detectAndComputeAsync(cuMat1g, cuQueryMask)
#     # _kps2, _descs2 = corb.detectAndComputeAsync(cuMat2g, cuTrainMask)
#     #
#     # # convert Keypoints to CPU
#     # kps1 = corb.convert(_kps1)
#     # kps2 = corb.convert(_kps2)
#     #
#     # # BruteForce Matching
#     # cbf = cv2.cuda_DescriptorMatcher.createBFMatcher(normType=cv2.NORM_HAMMING)
#     # cmatches = cbf.knnMatch(queryDescriptors=_descs1, trainDescriptors=_descs2, mask=None, k=2,
#     #                         compactResult=False)
#
#     good = []
#     pts1 = []
#     pts2 = []
#     if len(matches) > 1:
#         for m, n in matches:
#             if m.distance < 0.75 * n.distance:  # Ratio test
#                 query_kp = query_keypoints[m.queryIdx].pt
#                 train_kp = train_keypoints[m.trainIdx].pt
#
#                 if is_good_match(query_kp, train_kp, query_properties, train_properties, bb_pairs, query_array,
#                                  train_array):
#                     good.append([m])
#                     pts1.append(query_kp)
#                     pts2.append(train_kp)
#     else:
#         print("ValueError: not enough values to unpack (expected 2, got 1)")
#
#     ski_query_keypoints = cvkp_to_skikp(query_keypoints)
#     ski_train_keypoints = cvkp_to_skikp(train_keypoints)
#     ski_matches = cvmatch_to_skimatch(good)
#
#     pts1 = np.float32(pts1).reshape(-1, 1, 2)
#     pts2 = np.float32(pts2).reshape(-1, 1, 2)
#
#     cnt = 0
#     tc = [[0], [0], [0]]
#     Rc = [[0], [0], [0]]
#     try:
#         True
#     except:
#         print("An exception occurred")
#
#     init = None
#     while True:
#         model, inliers = ransac((ski_query_keypoints[ski_matches[:, 0]], ski_train_keypoints[ski_matches[:, 1]]),
#                                 FundamentalMatrixTransform, min_samples=8, residual_threshold=0.5, max_trials=5000,
#                                 initial_inliers=init)
#         # init = inliers
#
#         mask = np.ndarray(shape=(len(inliers), 1), dtype=int, order='F').astype(np.uint8)
#         for j, tmp in enumerate(inliers):
#             mask[j] = 1 if tmp else 0
#
#         # inlier_keypoints_left = ski_query_keypoints[ski_matches[inliers, 0]]
#         # inlier_keypoints_right = ski_train_keypoints[ski_matches[inliers, 1]]
#
#         K = np.array([[913.246621, 0.00000000, 1047.04624],
#                       [0.00000000, 1018.16544, 28.9628090],
#                       [0.00000000, 0.00000000, 1.00000000]])
#
#         Kt = K.transpose()
#         F = model.params
#         E = np.dot(np.dot(Kt, F), K)
#
#         _, R, t, mask = cv2.recoverPose(E=E, points1=pts1, points2=pts2, cameraMatrix=K, mask=mask)
#         R, _ = cv2.Rodrigues(R)
#
#         if t[2][0] > tc[2][0]:
#             tc = t
#             Rc = R
#
#         elapsed = (time.time() - start)
#
#         print(f'Number of matches: {ski_matches.shape[0]}')
#         print(f'Number of inliers: {inliers.sum()}')
#         print(f'Number of frames: {i}')
#         # print(f'Number of iterations: {cnt}')
#         print(f't[x]: {t[2]}')
#         print(f'elapsed: {timedelta(seconds=elapsed)}')
#         print("-----------------------------")
#
#         post_process_images(bb_pairs, query_img, train_img, query_properties, train_properties, False, query_array,
#                             train_array)
#
#         final_img = make_final_image(query_img, query_keypoints, train_img, train_keypoints, good, mask, i, matches, t,
#                                      R)
#
#         cv2.imshow("Matches", final_img)
#         cv2.imwrite("error.png", final_img)
#         cv2.waitKey()
#         cnt += 1
#         if t[2][0] > 0.995 or cnt >= 1:
#             break
#
#     side_slip = math.degrees(math.atan2(tc[0][0], tc[2][0]))
#     data_sideslip.append(side_slip)
#     data_iters.append(cnt)
#     data_frames.append(i)
#     data_tx.append(tc[2][0])
#     data_ty.append(tc[0][0])
#     data_tz.append(tc[1][0])
#     data_rx.append(Rc[2][0])
#     data_ry.append(Rc[0][0])
#     data_rz.append(Rc[1][0])
#     data_matches.append(len(matches))
#     data_good.append(len(good))
#     data_inliers.append(inliers.sum())
#     sum_tz += tc[1][0]
#     data_sumtz.append(sum_tz)
#
#     # if i >= 100:
#     #     break
#
# serialize_data(data_tx, data_ty, data_tz, data_rx, data_ry, data_rz, data_matches, data_good, data_inliers,
#                data_iters, data_frames, imu_rx, imu_ry, imu_rz, imu_packets)

data_tx, data_ty, data_tz, data_rx, data_ry, data_rz, data_matches, data_good, data_inliers, data_iters, data_frames, \
imu_rx, imu_ry, imu_rz, imu_packets \
    = deserialize_data("data-5000-0.5-cuda-nofilter")

w_data_tx, w_data_ty, w_data_tz, w_data_rx, w_data_ry, w_data_rz, w_data_matches, w_data_good, w_data_inliers, \
w_data_iters, w_data_frames, w_imu_rx, w_imu_ry, w_imu_rz, w_imu_packets \
    = deserialize_data("data-5000-0.5-cuda-nomask")

save_plotly_plots(data_tx, data_ty, data_tz, data_rx, data_ry, data_rz, data_matches, data_good, data_inliers,
                  data_iters, data_frames, imu_rx, imu_ry, imu_rz, imu_packets,
                  w_data_tx, w_data_ty, w_data_tz, w_data_rx, w_data_ry, w_data_rz, w_data_matches, w_data_good,
                  w_data_inliers, w_data_iters, w_data_frames, w_imu_rx, w_imu_ry, w_imu_rz, w_imu_packets)
