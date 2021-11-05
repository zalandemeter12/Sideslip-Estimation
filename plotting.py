import math
import pickle
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

RED = "#b31010"
LIGHT_RED = "#d17070"
GREEN = "#136927"
LIGHT_GREEN = "#71a57d"
BLUE = "#163078"
LIGHT_BLUE = "#7383ae"
ORANGE = "#fc7703"
DARK_ORANGE = "#974702"
LIGHT_ORANGE = "#fdad68"
LIGHT_GRAY = "#a6a6a6"
GRAY = "#525252"
YELLOW = "#cfb92d"

MID_RED = "#cf2d2d"
MID_GREEN = "#1d8236"
MID_BLUE = "#244e8a"
MID_ORANGE = "#ed9b02"

ANG_X = MID_RED
ANG_Y = MID_GREEN
ANG_Z = MID_BLUE

COLOR_SUM_RZ = MID_RED
COLOR_SUM_RZ_MASK = MID_GREEN
COLOR_SUM_RZ_IMU = MID_BLUE

COLOR_SUM_TZ = YELLOW
COLOR_SUM_TZ_MASK = MID_ORANGE

# COLOR_MATCH = "#ddaadd"
# COLOR_GOOD = "#8c74b2"
# COLOR_INLIER = "#2d4586"
COLOR_MATCH = MID_RED
COLOR_GOOD = MID_GREEN
COLOR_INLIER = MID_BLUE

# COLOR_ITER = "#c99097"
# COLOR_ITER_MASK = "#a54b62"
COLOR_ITER = YELLOW
COLOR_ITER_MASK = MID_ORANGE

def serialize_data(data_tx, data_ty, data_tz, data_rx, data_ry, data_rz, data_matches, data_good, data_inliers,
                   data_iters, data_frames, imu_rx, imu_ry, imu_rz, imu_packets):
    with open('data/data_tx', 'wb') as fp:
        pickle.dump(data_tx, fp)

    with open('data/data_ty', 'wb') as fp:
        pickle.dump(data_ty, fp)

    with open('data/data_tz', 'wb') as fp:
        pickle.dump(data_tz, fp)

    with open('data/data_rx', 'wb') as fp:
        pickle.dump(data_rx, fp)

    with open('data/data_ry', 'wb') as fp:
        pickle.dump(data_ry, fp)

    with open('data/data_rz', 'wb') as fp:
        pickle.dump(data_rz, fp)

    with open('data/data_matches', 'wb') as fp:
        pickle.dump(data_matches, fp)

    with open('data/data_good', 'wb') as fp:
        pickle.dump(data_good, fp)

    with open('data/data_inliers', 'wb') as fp:
        pickle.dump(data_inliers, fp)

    with open('data/data_iters', 'wb') as fp:
        pickle.dump(data_iters, fp)

    with open('data/data_frames', 'wb') as fp:
        pickle.dump(data_frames, fp)

    with open('data/imu_rx', 'wb') as fp:
        pickle.dump(imu_rx, fp)

    with open('data/imu_ry', 'wb') as fp:
        pickle.dump(imu_ry, fp)

    with open('data/imu_rz', 'wb') as fp:
        pickle.dump(imu_rz, fp)

    with open('data/imu_packets', 'wb') as fp:
        pickle.dump(imu_packets, fp)


def deserialize_data(folder):
    with open(f'{folder}/data_tx', 'rb') as fp:
        data_tx = pickle.load(fp)

    with open(f'{folder}/data_ty', 'rb') as fp:
        data_ty = pickle.load(fp)

    with open(f'{folder}/data_tz', 'rb') as fp:
        data_tz = pickle.load(fp)

    with open(f'{folder}/data_rx', 'rb') as fp:
        data_rx = pickle.load(fp)

    with open(f'{folder}/data_ry', 'rb') as fp:
        data_ry = pickle.load(fp)

    with open(f'{folder}/data_rz', 'rb') as fp:
        data_rz = pickle.load(fp)

    with open(f'{folder}/data_matches', 'rb') as fp:
        data_matches = pickle.load(fp)

    with open(f'{folder}/data_good', 'rb') as fp:
        data_good = pickle.load(fp)

    with open(f'{folder}/data_inliers', 'rb') as fp:
        data_inliers = pickle.load(fp)

    with open(f'{folder}/data_iters', 'rb') as fp:
        data_iters = pickle.load(fp)

    with open(f'{folder}/data_frames', 'rb') as fp:
        data_frames = pickle.load(fp)

    with open(f'{folder}/imu_rx', 'rb') as fp:
        imu_rx = pickle.load(fp)

    with open(f'{folder}/imu_ry', 'rb') as fp:
        imu_ry = pickle.load(fp)

    with open(f'{folder}/imu_rz', 'rb') as fp:
        imu_rz = pickle.load(fp)

    with open(f'{folder}/imu_packets', 'rb') as fp:
        imu_packets = pickle.load(fp)

    return data_tx, data_ty, data_tz, data_rx, data_ry, data_rz, data_matches, data_good, data_inliers, \
           data_iters, data_frames, imu_rx, imu_ry, imu_rz, imu_packets


def save_plots(data_tx, data_ty, data_tz, data_rx, data_ry, data_rz, data_matches, data_good, data_inliers,
               data_sumtz, data_iters, data_sideslip, data_frames):
    plt.plot(data_frames, data_tx)
    plt.xlabel('frames')
    plt.ylabel('t[x]')
    plt.title('Plot of the t[x] component')
    plt.savefig('imgs/tx.png', format='png', dpi=600)
    plt.close()

    plt.plot(data_frames, data_ty)
    plt.xlabel('frames')
    plt.ylabel('t[y]')
    plt.title('Plot of the t[y] component')
    plt.savefig('imgs/ty.png', format='png', dpi=600)
    plt.close()

    plt.plot(data_frames, data_tz)
    plt.xlabel('frames')
    plt.ylabel('t[z]')
    plt.title('Plot of the t[z] component')
    plt.savefig('imgs/tz.png', format='png', dpi=600)
    plt.close()

    plt.plot(data_frames, data_rx)
    plt.xlabel('frames')
    plt.ylabel('R[x]')
    plt.title('Plot of the R[x] component')
    plt.savefig('imgs/rx.png', format='png', dpi=600)
    plt.close()

    plt.plot(data_frames, data_ry)
    plt.xlabel('frames')
    plt.ylabel('R[y]')
    plt.title('Plot of the R[y] component')
    plt.savefig('imgs/ry.png', format='png', dpi=600)
    plt.close()

    plt.plot(data_frames, data_rz)
    plt.xlabel('frames')
    plt.ylabel('R[z]')
    plt.title('Plot of the R[z] component')
    plt.savefig('imgs/rz.png', format='png', dpi=600)
    plt.close()

    plt.plot(data_frames, data_matches)
    plt.xlabel('frames')
    plt.ylabel('all matches')
    plt.title('Plot of the number of all matches')
    plt.savefig('imgs/matches.png', format='png', dpi=600)
    plt.close()

    plt.plot(data_frames, data_good)
    plt.xlabel('frames')
    plt.ylabel('good matches')
    plt.title('Plot of the number of good matches')
    plt.savefig('imgs/good.png', format='png', dpi=600)
    plt.close()

    plt.plot(data_frames, data_inliers)
    plt.xlabel('frames')
    plt.ylabel('inliers')
    plt.title('Plot of the number of skimage inliers')
    plt.savefig('imgs/inliers.png', format='png', dpi=600)
    plt.close()

    plt.plot(data_frames, data_sumtz)
    plt.xlabel('frames')
    plt.ylabel('sum of tz')
    plt.title('Plot of the sum of tz component')
    plt.savefig('imgs/sumtz.png', format='png', dpi=600)
    plt.close()

    plt.plot(data_frames, data_iters)
    plt.xlabel('frames')
    plt.ylabel('iterations')
    plt.title('Plot of the required iterations')
    plt.savefig('imgs/iters.png', format='png', dpi=600)
    plt.close()

    plt.plot(data_frames, data_sideslip)
    plt.xlabel('frames')
    plt.ylabel('sideslip')
    plt.title('Plot of the sideslip')
    plt.savefig('imgs/sideslip.png', format='png', dpi=600)
    plt.close()


def pre_process_data(data_tx, data_ty, data_tz, data_rx, data_ry, data_rz, data_matches, data_good, data_inliers,
                     data_iters, data_frames, imu_rx, imu_ry, imu_rz, imu_packets):
    data_time = []
    for tmp in data_frames:
        data_time.append(tmp / 20)

    imu_time = []
    for tmp in imu_packets:
        imu_time.append(tmp / 40)

    ang_vel_x = []
    for i, tmp in enumerate(data_rx):
        ang_vel_x.append(tmp * 20)

    ang_vel_y = []
    for i, tmp in enumerate(data_ry):
        ang_vel_y.append(tmp * -1 * 20)
        data_ry[i] = data_ry[i] * -1

    ang_vel_z = []
    for i, tmp in enumerate(data_rz):
        ang_vel_z.append(tmp * -1 * 20)
        data_rz[i] = data_rz[i] * -1

    for i in range(len(data_ty)):
        data_ty[i] = data_ty[i] * -1  # invert axis

    for i in range(len(data_tz)):
        data_tz[i] = data_tz[i] * -1  # invert axis

    pd_ang_vel_x = pd.Series(ang_vel_x)
    pd_ang_vel_x = pd_ang_vel_x.rolling(10).mean()

    pd_ang_vel_y = pd.Series(ang_vel_y)
    pd_ang_vel_y = pd_ang_vel_y.rolling(10).mean()

    pd_ang_vel_z = pd.Series(ang_vel_z)
    pd_ang_vel_z = pd_ang_vel_z.rolling(10).mean()

    data_sideslip = []
    for i in range(len(data_tx)):
        side_slip = math.degrees(math.atan2(data_ty[i], data_tx[i]))
        data_sideslip.append(side_slip)

    pd_side = pd.Series(data_sideslip)
    pd_side = pd_side.rolling(20).mean()

    sum_tz_val = 0
    sum_tz = []
    for val in data_tz:
        sum_tz_val += val
        sum_tz.append(sum_tz_val)

    data_inliers_800 = []
    for i in range(len(data_inliers)):
        data_inliers_800.append(data_inliers[i] / 8000 + 0.9)

    sum_iters_val = 0
    sum_iters = []
    for val in data_iters:
        sum_iters_val += val
        sum_iters.append(sum_iters_val)

    sum_rz_val = 0
    sum_rz = []
    for val in data_rz:
        sum_rz_val += val
        sum_rz.append(sum_rz_val)

    imu_sum_rz_val = 0
    imu_sum_rz = []
    for val in imu_rz:
        imu_sum_rz_val += val
        imu_sum_rz.append(imu_sum_rz_val / 40)

    return data_tx, data_ty, data_tz, data_rx, data_ry, data_rz, data_matches, data_good, data_inliers, data_iters, \
           data_frames, imu_rx, imu_ry, imu_rz, imu_packets, data_time, pd_ang_vel_x, pd_ang_vel_y, pd_ang_vel_z, \
           pd_side, imu_time, data_sideslip, data_inliers_800, sum_iters, sum_rz, sum_tz, ang_vel_x, ang_vel_y, \
           ang_vel_z, imu_sum_rz


def save_plotly_plots(data_tx, data_ty, data_tz, data_rx, data_ry, data_rz, data_matches, data_good, data_inliers,
                      data_iters, data_frames, imu_rx, imu_ry, imu_rz, imu_packets,
                      w_data_tx, w_data_ty, w_data_tz, w_data_rx, w_data_ry, w_data_rz, w_data_matches, w_data_good,
                      w_data_inliers, w_data_iters, w_data_frames, w_imu_rx, w_imu_ry, w_imu_rz, w_imu_packets):

    data_tx, data_ty, data_tz, data_rx, data_ry, data_rz, data_matches, data_good, data_inliers, data_iters, \
    data_frames, imu_rx, imu_ry, imu_rz, imu_packets, data_time, pd_ang_vel_x, pd_ang_vel_y, pd_ang_vel_z, pd_side, \
    imu_time, data_sideslip, data_inliers_800, sum_iters, sum_rz, sum_tz, ang_vel_x, ang_vel_y, ang_vel_z, imu_sum_rz \
        = pre_process_data(data_tx, data_ty, data_tz, data_rx, data_ry, data_rz, data_matches, data_good, data_inliers,
                           data_iters, data_frames, imu_rx, imu_ry, imu_rz, imu_packets)

    w_data_tx, w_data_ty, w_data_tz, w_data_rx, w_data_ry, w_data_rz, w_data_matches, w_data_good, w_data_inliers, \
    w_data_iters, w_data_frames, w_imu_rx, w_imu_ry, w_imu_rz, w_imu_packets, w_data_time, w_pd_ang_vel_x, \
    w_pd_ang_vel_y, w_pd_ang_vel_z, w_pd_side, w_imu_time, w_data_sideslip, w_data_inliers_800, w_sum_iters, w_sum_rz, \
    w_sum_tz, w_ang_vel_x, w_ang_vel_y, w_ang_vel_z, w_imu_sum_rz \
        = pre_process_data(w_data_tx, w_data_ty, w_data_tz, w_data_rx, w_data_ry, w_data_rz, w_data_matches,
                           w_data_good, w_data_inliers, w_data_iters, w_data_frames, w_imu_rx, w_imu_ry, w_imu_rz,
                           w_imu_packets)

    # ==================================================================
    # ANUGLAR VELOCITY
    # ==================================================================

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Raw estimate", "Rolling mean estimate, window size 10", "IMU data"))

    fig.add_trace(go.Scatter(x=data_time, y=ang_vel_x, name='X', line=dict(color=ANG_X, width=2),
                             legendgroup='1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=ang_vel_y, name='Y', line=dict(color=ANG_Y, width=2),
                             legendgroup='1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=ang_vel_z, name='Z', line=dict(color=ANG_Z, width=2),
                             legendgroup='1'), row=1, col=1)

    fig.add_trace(go.Scatter(x=data_time, y=pd_ang_vel_x, name='X', line=dict(color=ANG_X, width=2),
                             legendgroup='2', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=pd_ang_vel_y, name='Y', line=dict(color=ANG_Y, width=2),
                             legendgroup='2', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=pd_ang_vel_z, name='Z', line=dict(color=ANG_Z, width=2),
                             legendgroup='2', showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=imu_time, y=imu_rx, name='X', line=dict(color=ANG_X, width=2),
                             legendgroup='3', showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=imu_time, y=imu_ry, name='Y', line=dict(color=ANG_Y, width=2),
                             legendgroup='3', showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=imu_time, y=imu_rz, name='Z', line=dict(color=ANG_Z, width=2),
                             legendgroup='3', showlegend=False), row=3, col=1)

    # Edit the layout
    fig.update_layout(title='Comparing angular velocity estimate to IMU data',
                      title_x=0.5,
                      xaxis1_title='Time [s]',
                      yaxis1_title='Angular velocity [rad/s]',
                      yaxis1_range = [-1, 1.8],
                      xaxis2_title='Time [s]',
                      yaxis2_title='Angular velocity [rad/s]',
                      yaxis2_range=[-1, 1.65],
                      xaxis3_title='Time [s]',
                      yaxis3_title='Angular velocity [rad/s]',
                      yaxis3_range=[-1, 1.3],
                      )

    fig.write_image("imgs/angular_velocity.png", format="png", height=720, width=900, scale=4)

    # ==================================================================
    # ANUGLAR VELOCITY 2
    # ==================================================================

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("X component", "Y component", "Z component"),
                        specs=[[{"colspan": 2}, None],
                               [{}, {}]],
                        )

    fig.add_trace(go.Scatter(x=imu_time, y=imu_rx, name='X - IMU', line=dict(color=LIGHT_RED, width=2),
                             legendgroup='1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=pd_ang_vel_x, name='X - AVG', line=dict(color=RED, width=2),
                             legendgroup='1'), row=1, col=1)


    fig.add_trace(go.Scatter(x=imu_time, y=imu_ry, name='Y - IMU', line=dict(color=LIGHT_GREEN, width=2),
                             legendgroup='2'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=pd_ang_vel_y, name='Y- AVG', line=dict(color=GREEN, width=2),
                             legendgroup='2'), row=2, col=1)


    fig.add_trace(go.Scatter(x=imu_time, y=imu_rz, name='Z - IMU', line=dict(color=LIGHT_BLUE, width=2),
                             legendgroup='3'), row=2, col=2)
    fig.add_trace(go.Scatter(x=data_time, y=pd_ang_vel_z, name='Z- AVG', line=dict(color=BLUE, width=2),
                             legendgroup='3'), row=2, col=2)


    # Edit the layout
    fig.update_layout(title='Comparing angular velocity estimate to IMU data',
                      title_x=0.5,
                      xaxis1_title='Time [s]',
                      yaxis1_title='Angular velocity [rad/s]',
                      xaxis2_title='Time [s]',
                      yaxis2_title='Angular velocity [rad/s]',
                      xaxis3_title='Time [s]',
                      yaxis3_title='Angular velocity [rad/s]'
                      )

    fig.write_image("imgs/angular_velocity-2.png", format="png", height=720, width=900, scale=4)


    # ==================================================================
    # SIDESLIP
    # ==================================================================

    lower_limit = []
    upper_limit = []
    for i in range(len(data_sideslip)):
        lower_limit.append(-5.73)
        upper_limit.append(5.73)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("With masked features", "Without masked features")
                        )

    fig.add_trace(go.Scatter(x=data_time, y=data_sideslip, name='raw', line=dict(color=LIGHT_RED, width=2),
                             legendgroup='1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=pd_side, name='average', line=dict(color=RED, width=2),
                             legendgroup='1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=upper_limit, name='upper limit', line=dict(color=GRAY, width=2,
                                                                                       dash='dash'),
                             legendgroup='1', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=lower_limit, name='lower limit', line=dict(color=GRAY, width=2,
                                                                                       dash='dash'),
                             legendgroup='1', showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=data_time, y=w_data_sideslip, name='raw', line=dict(color=LIGHT_ORANGE, width=2),
                             legendgroup='2'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=w_pd_side, name='average', line=dict(color=ORANGE, width=2),
                             legendgroup='2'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=upper_limit, name='upper limit', line=dict(color=GRAY, width=2,
                                                                                       dash='dash'),
                             legendgroup='2', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=lower_limit, name='lower limit', line=dict(color=GRAY, width=2,
                                                                                       dash='dash'),
                             legendgroup='2', showlegend=False), row=2, col=1)

    fig.update_layout(title='Sideslip angle estimate with rolling mean average, window size 20',
                      title_x=0.5,
                      xaxis1_title='Time [s]',
                      yaxis1_title='Sideslip angle [deg]',
                      xaxis2_title='Time [s]',
                      yaxis2_title='Sideslip angle [deg]',
                      legend_tracegroupgap=300
                      )

    fig.update_yaxes(range=[-10, 10])

    fig.write_image("imgs/sideslip.png", format="png", height=720, width=1080, scale=4)

    # # ==================================================================
    # # TRANSLATION - X
    # # ==================================================================
    #
    # fig = go.Figure()
    #
    # fig.add_trace(go.Scatter(x=data_time, y=w_data_tx, name='X', line=dict(color=LIGHT_RED, width=2)))
    # fig.add_trace(go.Scatter(x=data_time, y=data_tx, name='X - masked', line=dict(color=RED, width=2)))
    #
    # fig.update_layout(title='Normalized displacement estimate, X component',
    #                   title_x=0.5,
    #                   xaxis_title='Time [s]',
    #                   yaxis_title='Normalized displacement [-]',
    #                   plot_bgcolor=BGCOLOR
    #                   )
    #
    # fig.write_image("imgs/tx.png", format="png", height=720, width=1080, scale=4)
    #
    # # ==================================================================
    # # TRANSLATION - Y
    # # ==================================================================
    #
    # fig = go.Figure()
    #
    # fig.add_trace(go.Scatter(x=data_time, y=w_data_ty, name='Y', line=dict(color=LIGHT_GREEN, width=2)))
    # fig.add_trace(go.Scatter(x=data_time, y=data_ty, name='Y - masked', line=dict(color=GREEN, width=2)))
    #
    # fig.update_layout(title='Normalized displacement estimate, Y component',
    #                   title_x=0.5,
    #                   xaxis_title='Time [s]',
    #                   yaxis_title='Normalized displacement [-]',
    #                   plot_bgcolor=BGCOLOR
    #                   )
    #
    # fig.write_image("imgs/ty.png", format="png", height=720, width=1080, scale=4)
    #
    # # ==================================================================
    # # TRANSLATION - Z
    # # ==================================================================
    #
    # fig = go.Figure()
    #
    # fig.add_trace(go.Scatter(x=data_time, y=w_data_tz, name='Z', line=dict(color=LIGHT_BLUE, width=2)))
    # fig.add_trace(go.Scatter(x=data_time, y=data_tz, name='Z - masked', line=dict(color=BLUE, width=2)))
    #
    # fig.update_layout(title='Normalized displacement estimate, Z component',
    #                   title_x=0.5,
    #                   xaxis_title='Time [s]',
    #                   yaxis_title='Normalized displacement [-]',
    #                   plot_bgcolor=BGCOLOR
    #                   )
    #
    # fig.write_image("imgs/tz.png", format="png", height=720, width=1080, scale=4)

    # ==================================================================
    # TRANSLATION - XYZ
    # ==================================================================

    lower_limit = []
    for i in range(len(sum_tz)):
        lower_limit.append(0.995)

    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=("X component", "X component", "Y component", "Z component"),
                        specs=[[{"colspan": 2}, None],
                               [{"colspan": 2}, None],
                               [{}, {}]],
                        )

    fig.add_trace(go.Scatter(x=data_time, y=w_data_tx, name='X', line=dict(color=LIGHT_RED, width=2),
                             legendgroup='1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=data_tx, name='X - masked', line=dict(color=RED, width=2),
                             legendgroup='1'), row=1, col=1)

    fig.add_trace(go.Scatter(x=data_time, y=w_data_tx, name='X', line=dict(color=LIGHT_RED, width=2),
                             legendgroup='2'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=data_tx, name='X - masked', line=dict(color=RED, width=2),
                             legendgroup='2'), row=2, col=1)

    fig.add_trace(go.Scatter(x=data_time, y=lower_limit, name='threshold', line=dict(color=GRAY, width=2,
                                                                                       dash='dash'),
                             legendgroup='2'), row=2, col=1)

    fig.add_trace(go.Scatter(x=data_time, y=w_data_ty, name='Y', line=dict(color=LIGHT_GREEN, width=2),
                             legendgroup='3'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=data_ty, name='Y - masked', line=dict(color=GREEN, width=2),
                             legendgroup='3'), row=3, col=1)

    fig.add_trace(go.Scatter(x=data_time, y=w_data_tz, name='Z', line=dict(color=LIGHT_BLUE, width=2),
                             legendgroup='3'), row=3, col=2)
    fig.add_trace(go.Scatter(x=data_time, y=data_tz, name='Z - masked', line=dict(color=BLUE, width=2),
                             legendgroup='3'), row=3, col=2)

    # Edit the layout
    fig.update_layout(title='Normalized displacement estimate',
                      title_x=0.5,
                      xaxis1_title='Time [s]',
                      yaxis1_title='Normalized displacement [-]',
                      yaxis1_range=[0.85, 1],
                      xaxis2_title='Time [s]',
                      yaxis2_title='Normalized displacement [-]',
                      yaxis2_range=[0.99, 1],
                      xaxis3_title='Time [s]',
                      yaxis3_title='Normalized displacement [-]',
                      xaxis4_title='Time [s]',
                      yaxis4_title='Normalized displacement [-]',
                      legend_tracegroupgap=240,
                      )

    fig.write_image("imgs/t_xyz.png", format="png", height=720, width=900, scale=4)

    # ==================================================================
    # ITERATIONS
    # ==================================================================

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Without masked features", "With masked features", "Comparing sums")
                        )

    fig.add_trace(go.Scatter(x=data_time, y=w_data_iters, name='iters', line=dict(color=COLOR_ITER, width=2),
                             legendgroup='0', showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=data_time, y=data_iters, name='iters - masked', line=dict(color=COLOR_ITER_MASK, width=2),
                             legendgroup='0', showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=data_time, y=w_sum_iters, name='iters', line=dict(color=COLOR_ITER, width=2),
                             legendgroup='1'), row=3, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=sum_iters, name='iters - masked', line=dict(color=COLOR_ITER_MASK, width=2),
                             legendgroup='1'), row=3, col=1)

    # Edit the layout
    fig.update_layout(title='Iterations needed to surpass threshold',
                      title_x=0.5,
                      xaxis1_title='Time [s]',
                      yaxis1_title='Iterations needed [-]',
                      xaxis2_title='Time [s]',
                      yaxis2_title='Iterations needed [-]',
                      xaxis3_title='Time [s]',
                      yaxis3_title='Sum of iterations needed [-]'
                      # legend_tracegroupgap=50,
                      )

    fig.write_image("imgs/iters.png", format="png", height=720, width=900, scale=4)

    # ==================================================================
    # MATCHES-GOOD-INLIERS
    # ==================================================================

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("All matches", "Good matches and inliers")
                        )

    fig.add_trace(go.Scatter(x=data_time, y=data_matches, name='all matches', line=dict(color=COLOR_MATCH, width=2),
                             legendgroup='1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=data_good, name='good matches', line=dict(color=COLOR_GOOD, width=2),
                             legendgroup='2'), row=2, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=data_inliers, name='inliers', line=dict(color=COLOR_INLIER, width=2),
                             legendgroup='2'), row=2, col=1)

    # Edit the layout
    fig.update_layout(title='Number of matches found and used',
                      title_x=0.5,
                      xaxis1_title='Time [s]',
                      yaxis1_title='Number of matches [-]',
                      xaxis2_title='Time [s]',
                      yaxis2_title='Number of matches [-]',
                      legend_tracegroupgap=320,
                      )

    fig.write_image("imgs/matches.png", format="png", height=720, width=900, scale=4)

    # ==================================================================
    # SUM TZ - RZ
    # ==================================================================

    upper_limit = []
    for i in range(len(sum_rz)):
        upper_limit.append(6.28318531)

    lower_limit = []
    for i in range(len(sum_tz)):
        lower_limit.append(0)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Translation", "Rotation"),
                        )

    fig.add_trace(go.Scatter(x=data_time, y=w_sum_tz, name='tz', line=dict(color=COLOR_SUM_TZ, width=2),
                             legendgroup='1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=sum_tz, name='tz - masked', line=dict(color=COLOR_SUM_TZ_MASK, width=2),
                             legendgroup='1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data_time, y=lower_limit, name='lower limit', line=dict(color=GRAY, width=2,
                                                                                       dash='dash'),
                             legendgroup='1', showlegend=False), row=1, col=1)

    fig.add_trace(go.Scatter(x=data_time, y=w_sum_rz, name='Rz', line=dict(color=COLOR_SUM_RZ, width=2),
                             legendgroup='2'), row=1, col=2)
    fig.add_trace(go.Scatter(x=data_time, y=sum_rz, name='Rz - masked', line=dict(color=COLOR_SUM_RZ_MASK, width=2),
                             legendgroup='2'), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=imu_time, y=imu_sum_rz, name='IMU', line=dict(color=COLOR_SUM_RZ_IMU, width=2),
                   legendgroup='2'), row=1, col=2)
    fig.add_trace(go.Scatter(x=data_time, y=upper_limit, name='upper limit', line=dict(color=GRAY, width=2,
                                                                                       dash='dash'),
                             legendgroup='1', showlegend=False), row=1, col=2)

    fig.update_layout(title='Sums of the translation and rotation Z component',
                      title_x=0.5,
                      xaxis1_title='Time [s]',
                      yaxis1_title='Normalized sum of tz [-]',
                      xaxis2_title='Time [s]',
                      yaxis2_title='Sum of Rz [rad]'
                      )

    fig.write_image("imgs/sums.png", format="png", height=640, width=1366, scale=4)
