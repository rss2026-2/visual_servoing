import rosbag2_py
import matplotlib.pyplot as plt
import os

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def extract(bagfile, topic, start_s=0.0, end_s=None):
    bagfile = os.path.expanduser(bagfile)
    bag_uri = os.path.dirname(bagfile) if bagfile.endswith(".db3") else bagfile

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_uri, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    msg_type = None
    for t in reader.get_all_topics_and_types():
        if t.name == topic:
            msg_type = get_message(t.type)
            break

    if msg_type is None:
        return [], [], [], []

    ts0 = None
    times = []
    distance_error = []
    x_error = []
    y_error = []

    while reader.has_next():
        topic_name, data, timestamp_ns = reader.read_next()
        if topic_name != topic:
            continue

        if ts0 is None:
            ts0 = timestamp_ns

        t_s = (timestamp_ns - ts0) * 1e-9
        if t_s < start_s:
            continue
        if end_s is not None and t_s > end_s:
            break

        times.append(t_s - start_s)

        msg = deserialize_message(data, msg_type)
        distance_error.append(msg.distance_error)
        x_error.append(msg.x_error)
        y_error.append(msg.y_error)

    return times, distance_error, x_error, y_error


# filepath
bags = [
    "~/bags/far_right_1",
    "~/bags/far_right_2",
    "~/bags/far_right_3",
]

# the timeframe for each run (normalize the starts of the bags)
windows_s = {
    "far_right_1": (1.0, 10.0),
    "far_right_2": (1.0,10.0)
}

fig, (ax_d, ax_x, ax_y) = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
fig.suptitle("Errors for Parking When Cone is Far Away and to the Right", y=0.995)

for i, b in enumerate(bags, start=1):
    name = os.path.basename(os.path.expanduser(b).rstrip("/"))
    start_s, end_s = windows_s.get(name, (0.0, None))
    t, d, x, y = extract(b, "/parking_error", start_s=start_s, end_s=end_s)

    ax_d.plot(t, d, label=f"Trial {i}")
    ax_x.plot(t, x, label=f"Trial {i}")
    ax_y.plot(t, y, label=f"Trial {i}")

ax_d.set_ylabel("Distance error")
ax_x.set_ylabel("X error")
ax_y.set_ylabel("Y error")
ax_y.set_xlabel("Time (s)")

ax_d.legend()
plt.tight_layout()
plt.savefig("parking_errors.png", dpi=200)
plt.close()
