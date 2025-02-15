import argparse
import cv2
import json
import numpy as np
import sys
import torch

import aria.sdk as aria

from common import quit_keypress, update_iptables
from gaze_model.inference import infer
from projectaria_tools.core.mps import EyeGaze
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    get_linear_camera_calibration,
    distort_by_calibration,  # NEW: Import the distortion function.
)
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core.sophus import SE3

from write_frame import *

# File paths to model weights and configuration
model_weights = "gaze_model/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
model_config = "gaze_model/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
model_device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--device-ip",
        default=None,
        type=str,
        help="Set glasses IP address for connection",
    )
    parser.add_argument(
        "--post-process",
        action="store_true",
        default=False,
        help="Toggle on post processing in recording (will save to a separate mp4)",
    )
    return parser.parse_args()


def gaze_inference(
    data: np.ndarray,
    inference_model,
    rgb_stream_label,
    device_calibration,
    rgb_camera_calibration,
):
    depth_m = 1  # 1 m

    # Prepare the image tensor for inference.
    img = torch.tensor(data, device="cuda")

    with torch.no_grad():
        preds, lower, upper = inference_model.predict(img)
        preds = preds.detach().cpu().numpy()
        lower = lower.detach().cpu().numpy()
        upper = upper.detach().cpu().numpy()

    eye_gaze = EyeGaze
    eye_gaze.yaw = preds[0][0]
    eye_gaze.pitch = preds[0][1]

    # Compute the eye gaze vector at depth_m reprojection in the image.
    gaze_projection = get_gaze_vector_reprojection(
        eye_gaze,
        rgb_stream_label,
        device_calibration,
        rgb_camera_calibration,
        depth_m,
    )

    # Adjust for image rotation
    width = 1408
    if gaze_projection is None or not np.any(gaze_projection):
        return (0, 0)
    x, y = gaze_projection
    rotated_x = width - y
    rotated_y = x

    return (rotated_x, rotated_y)


def display_text(image, text: str, position, color=(0, 0, 255)):
    cv2.putText(
        img=image,
        text=text,
        org=position,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=color,
        thickness=3,
    )


def main():
    global data
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()
    print(args.post_process)

    # Initialize model for inference.
    model = infer.EyeGazeInference(model_weights, model_config, model_device)

    # Set up the custom data recorder.
    data = DataRecorder(frame_name="gaze_data", coord_name="gaze_data", framerate=10, post=args.post_process)

    # Optional: Set SDK's log level.
    aria.set_log_level(aria.Level.Info)

    # 1. Create and connect the DeviceClient.
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)
    device = device_client.connect()

    # 2. Create StreamingClient instance and retrieve the streaming_manager.
    streaming_manager = device.streaming_manager
    streaming_client = aria.StreamingClient()

    # 3. Configure subscription to listen to Aria's RGB and EyeTrack streams.
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1

    # Set security options.
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    # 4. Create and attach observer to the streaming client.
    class StreamingClientObserver:
        def __init__(self):
            self.images = {}

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.images[record.camera_id] = image

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    # 5. Start listening.
    print("Start listening to image data")
    streaming_client.subscribe()

    # 6. Visualize the streaming data.
    rgb_window = "Aria RGB"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 1024, 1024)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    # 7. Fetch calibration and labels.
    rgb_stream_label = "camera-rgb"
    device_calib_str = streaming_manager.sensors_calibration()

    # Parse JSON to extract RGB camera calibration for gaze inference.
    parser = json.loads(device_calib_str)
    rgb_camera_calib_json = next(
        camera for camera in parser["CameraCalibrations"] if camera["Label"] == "camera-rgb"
    )

    # Extract and preprocess translation and quaternion.
    translation = rgb_camera_calib_json["T_Device_Camera"]["Translation"]
    quaternion = rgb_camera_calib_json["T_Device_Camera"]["UnitQuaternion"]
    quat_w = quaternion[0]
    quat = np.array(quaternion[1]).reshape(3, 1)
    translation = np.array(translation).reshape(3, 1)

    # Create the SE3 transform.
    se3_transform = SE3.from_quat_and_translation(quat_w, quat, translation)

    # Compute the linear camera calibration used for gaze reprojection.
    rgb_camera_calibration = get_linear_camera_calibration(1408, 1408, 550, "camera-rgb", se3_transform)

    # Convert the device calibration from JSON string to DeviceCalibration object.
    device_calibration = device_calibration_from_json_string(device_calib_str)

    # For undistortion, get the original RGB calibration from the device calibration.
    rgb_calib = device_calibration.get_camera_calib("camera-rgb")
    # Compute the destination (linear) calibration for undistortion.
    dst_calib = get_linear_camera_calibration(1408, 1408, 550, "camera-rgb")

    np.set_printoptions(threshold=np.inf)

    # 8. Main loop: process each frame.
    while not quit_keypress():
        try:
            if aria.CameraId.Rgb in observer.images:
                # Get the RGB image, rotate, and convert its color space.
                rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                rgb_image = distort_by_calibration(rgb_image, dst_calib, rgb_calib)

                gaze = observer.images.get(aria.CameraId.EyeTrack)

                if gaze is not None and np.mean(gaze) > 10 and np.median(gaze) > 10:
                    # Run gaze inference.
                    gaze_coordinates = gaze_inference(
                        gaze, model, rgb_stream_label, device_calibration, rgb_camera_calibration
                    )

                    if gaze_coordinates is not None:
                        data.record_frame(rgb_image, (int(gaze_coordinates[0]), int(gaze_coordinates[1])))
                        cv2.circle(
                            rgb_image,
                            (int(gaze_coordinates[0]), int(gaze_coordinates[1])),
                            5,
                            (0, 255, 0),
                            10,
                        )

                    # Log the gaze coordinates.
                    display_text(
                        rgb_image,
                        f'Gaze Coordinates: ({round(gaze_coordinates[0], 4)}, {round(gaze_coordinates[1], 4)})',
                        (20, 90),
                    )
                else:
                    if args.post_process:
                        data.record_frame(rgb_image, (-1, -1))
                    display_text(rgb_image, "No Gaze Found", (20, 50))

                # Show the original RGB image.
                cv2.imshow(rgb_window, rgb_image)

                if args.post_process:
                    data.record_frame_post(rgb_image)

                del observer.images[aria.CameraId.Rgb]
        except Exception as e:
            print(f"Encountered error: {e}")

    # 9. Unsubscribe to clean up resources.
    print("Stop listening to image data")
    streaming_client.unsubscribe()


if __name__ == "__main__":
    try:
        main()
    finally:
        data.end_recording()
