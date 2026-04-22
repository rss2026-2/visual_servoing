import rosbag2_py
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def extract(bag_path, topic, ignore_first_sec=0, max_time=None):

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_type = get_message(topic_types[topic])

    t, x,y,dist = [], [], [], []

    while reader.has_next():
        topic_name, data, ts = reader.read_next()

        if topic_name != topic:
            continue

        msg = deserialize_message(data, msg_type)

        t.append(ts / 1e9)
        x.append(msg.x_error)
        y.append(msg.y_error)
        dist.append(msg.distance_error)

    if not t:
        return [], []

    t0 = t[0]
    print(f'{t0=}')
    t = [x - t0 for x in t]

    # ignore first x seconds
    filtered_t = []
    filtered_x = []
    filtered_y = []
    filtered_d = []

    for ti, xi, yi, di in zip(t, x, y, dist):
        if ti < ignore_first_sec:
            continue
        if max_time is not None and ti -ignore_first_sec > max_time:
            break   # safe since time is increasing
        filtered_t.append(ti - ignore_first_sec)
        filtered_y.append(yi)
        filtered_x.append(xi)
        filtered_d.append(di)

    return filtered_t, [filtered_x, filtered_y, filtered_d]


bags = [
    # "lab4_rosbags/homography_tests/testx_dyn_1",
    # "lab4_rosbags/homography_tests/testx_dyn_2",
    # "lab4_rosbags/homography_tests/testx_dyn_3",
    # "lab4_rosbags/wf_speedtest/wf_speedtest3_real_trial1"
    "lab4_rosbags/cone_tests/too_close_straight_1",
    "lab4_rosbags/cone_tests/too_close_straight_2",
    "lab4_rosbags/cone_tests/too_close_straight_3",
    "lab4_rosbags/cone_tests/too_close_straight_4",


]

ignore_first_seconds_ = [0, 0, 0, 0, ]

all_v = []

# for b, i in zip(bags, ignore_first_seconds_):
#     t, v = extract(b, "/observed_error", ignore_first_sec = i, max_time=13)
#     all_v.append(v)


#     legend_label = "Trial " + b[-1]
#     if "45" in b:
#         plt.plot(t, v, label=" $\\frac{\pi}{4}$ rad " + legend_label, color='green')
#     else :
#         plt.plot(t, v, label="0 rad " + legend_label, color='blue', linestyle="-")

# for b, k in zip(bags, ignore_first_seconds_):
#     t, v = extract(b, "/vesc/low_level/input/safety", ignore_first_sec=k)
#     all_v.append(v)

# # v_straight = [(a + b +c)/3 for a,b,c in zip(*all_v[:3])]
# # v_angled = [(x+y+z)/3 for x,y,z in zip(*all_v[3:])]
# # plt.plot(t, v_straight, label="$\\frac{\pi}{4}$ rad", color='green')
# # plt.plot(t, v_angled, label="0 rad", color='blue')

#     steering_rate = []
#     rate_time = []

#     for i in range(1, len(v)):
#         dv = v[i] - v[i-1]
#         dt = t[i] - t[i-1]

#         steering_rate.append(dv / dt)
#         rate_time.append(t[i])

#     plt.plot(rate_time, steering_rate, label=b)
for b, ignore in zip(bags, ignore_first_seconds_):

    t, vals = extract(b, "/parking_error", ignore_first_sec=ignore)

    # steering_rate = []
    # rate_time = []

    # for k in range(1, len(v)):
    #     dv = v[k] - v[k-1]
    #     dt = t[k] - t[k-1]

    #     steering_rate.append(dv / dt)
    #     rate_time.append(t[k])
    labels =     ["X", "Y", "Distance"]
    for v, l in zip(vals, labels):
        plt.plot(t, v, label=l)

    plt.xlabel("Time (s)")
    plt.ylabel("Parking Error Error (m)")
    plt.title("Parking Error Trial " + str(b[-1]))
    plt.legend()
    plt.savefig("Dyn hom error" + str(b[-1]) + ".png")
    plt.show()
    plt.close()
