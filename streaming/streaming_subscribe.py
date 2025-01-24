# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from projectaria_tools.core.calibration import device_calibration_from_json_string, get_linear_camera_calibration
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core.sophus import SE3

# file paths to model weights and configuration
model_weights = f"gaze_model/inference/model/pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
model_config = f"gaze_model/inference/model/pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
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
    return parser.parse_args()

def gaze_inference(data: np.ndarray, inference_model, rgb_stream_label, device_calibration, rgb_camera_calibration):
    depth_m = 1  # 1 m

    # Iterate over the data and LOG data as we see fit
    img = torch.tensor(
        data, device="cuda"
    )

    with torch.no_grad():
        preds, lower, upper = inference_model.predict(img)
        preds = preds.detach().cpu().numpy()
        lower = lower.detach().cpu().numpy()
        upper = upper.detach().cpu().numpy()

    eye_gaze = EyeGaze
    eye_gaze.yaw = preds[0][0]
    eye_gaze.pitch = preds[0][1] 

    # Compute eye_gaze vector at depth_m reprojection in the image
    gaze_projection = get_gaze_vector_reprojection(
        eye_gaze,
        rgb_stream_label,
        device_calibration,
        rgb_camera_calibration,
        depth_m,
    )

    # Adjust for image rotation
    width = 1408
    if gaze_projection.any() is None:
        return (0, 0)
    x, y = gaze_projection
    rotated_x = width - y
    rotated_y = x

    return (rotated_x, rotated_y)

def display_text(image, text: str, position, color=(0, 0, 255)):
    cv2.putText(
        img = image,
        text = text,
        org = position,
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        color = color,
        thickness = 3
    )

def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    # Initialize model for inference using path to model weight and config file
    model = infer.EyeGazeInference(
        model_weights,
        model_config,
        model_device
    )

    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    # 1. Create and connect the DeviceClient for fetching device calibration, which is required to cast 3D gaze prediction to 2D image
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)
    device = device_client.connect()

    # 2. Create StreamingClient instance and retrieve streaming_manager
    streaming_manager = device.streaming_manager
    streaming_client = aria.StreamingClient()

    # 3. Configure subscription to listen to Aria's RGB streams.
    # @see StreamingDataType for the other data types
    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack
    )

    # A shorter queue size may be useful if the processing callback is always slow and you wish to process more recent data
    # For visualizing the images, we only need the most recent frame so set the queue size to 1
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1

    # Set the security options
    # @note we need to specify the use of ephemeral certs as this sample app assumes
    # aria-cli was started using the --use-ephemeral-certs flag
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    # 4. Create and attach observer to streaming client
    class StreamingClientObserver:
        def __init__(self):
            self.images = {}

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.images[record.camera_id] = image

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    # 5. Start listening
    print("Start listening to image data")
    streaming_client.subscribe()

    # 6. Visualize the streaming data until we close the window
    #    Live gaze stream will be cast onto the RGB window
    rgb_window = "Aria RGB"

    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 1024, 1024)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    # 7. Fetch calibration and labels to be passed to 3D -> 2D gaze coordinate casting function
    rgb_stream_label = "camera-rgb"
    device_calibration = streaming_manager.sensors_calibration()

    # sensors_calibration() returns device calibration in JSON format, so we must parse through in order to find the calibration for the RGB camera
    parser = json.loads(device_calibration)
    rgb_camera_calibration = next(
        camera for camera in parser['CameraCalibrations'] 
        if camera['Label'] == 'camera-rgb'
    )

    # Convert device calibration from JSON string to DeviceCalibration Object
    device_calibration = device_calibration_from_json_string(device_calibration)

    # Extract translation and quaternion variables from camera calibration JSON and preprocess
    translation = rgb_camera_calibration["T_Device_Camera"]["Translation"]
    quaternion = rgb_camera_calibration["T_Device_Camera"]["UnitQuaternion"]
    # quaternion format is [w, [x, y, z]]
    quat_w = quaternion[0]
    quat =[quaternion[1][0], quaternion[1][1], quaternion[1][2]]
    # convert both to numpy arrays with shape (3, 1)
    quat = np.array(quat).reshape(3, 1)
    translation = np.array(translation).reshape(3, 1)

    # Create SE3 Object containing information on quaternion coordinate and translations
    se3_transform = SE3.from_quat_and_translation(quat_w, quat, translation)

    # Retrieve RGB camera calibration from SE3 Object
    # The focal length can also be 611.1120152034575
    rgb_camera_calibration = get_linear_camera_calibration(1408, 1408, 550, 'camera-rgb', se3_transform) # the dimensions of the RGB camera is (1408, 1408)

    np.set_printoptions(threshold=np.inf) # set print limit to inf

    # 8. Continuously loop through and run gaze estimation + postprocessing on every frame before displaying
    while not quit_keypress():
        # Render the RGB image
        try:
            if aria.CameraId.Rgb in observer.images:
                rgb_image = np.rot90(observer.images[aria.CameraId.Rgb], -1)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

                gaze = observer.images.get(aria.CameraId.EyeTrack)
                
                if gaze is not None and np.mean(gaze) > 10 and np.median(gaze) > 10:
                    
                    # Run inference using gaze estimation model
                    gaze_coordinates = gaze_inference(gaze, model, rgb_stream_label, device_calibration, rgb_camera_calibration)

                    # If gaze coordinates exist, plot as a bright green dot on screen
                    if gaze_coordinates is not None:
                        cv2.circle(rgb_image, (int(gaze_coordinates[0]), int(gaze_coordinates[1])), 5, (0, 255, 0), 10)
                    
                    # Log coordinates of gaze with text
                    display_text(rgb_image, f'Gaze Coordinates: ({round(gaze_coordinates[0], 4)}, {round(gaze_coordinates[1], 4)})', (20, 90))

                else:
                    display_text(rgb_image, 'No Gaze Found', (20, 50))

                cv2.imshow(rgb_window, rgb_image)
                del observer.images[aria.CameraId.Rgb]
        except Exception as e:
            print(f'Encountered error: {e}')
        
    # 9. Unsubscribe to clean up resources
    print("Stop listening to image data")
    streaming_client.unsubscribe()

if __name__ == "__main__":
    main()
