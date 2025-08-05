
import cv2
import ffmpeg
import numpy as np
import time
from datetime import datetime

class DataRecorder():
    def __init__(self, frame_name="gaze_data", coord_name="gaze_data", framerate=10, post=False, record_timestamp=True):
        recording_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        self.output_file_frame = frame_name + str(recording_id) + ".mp4"
        self.output_file_gaze = coord_name + str(recording_id) + ".npy"
        self.output_file_timestamps = coord_name + "_timestamps_" + str(recording_id) + ".npy"
        self.output_np_gaze = []
        self.record_timestamp = record_timestamp
        self.timestamps = []
        self.framerate = int(framerate)
        self.post = post

        # Get frame dimensions from the first image. Assumes all are same size
        self.frame_height, self.frame_width, _ = 1408, 1408, 3 # aria glasses outputs are 1408x1408x3 unless you explicitely change it

        # Setup the ffmpeg pipe for video writing
        self.process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(self.frame_width, self.frame_height), framerate = self.framerate)
            .output(self.output_file_frame, vcodec='libx264', pix_fmt='yuv420p') # Using h264 codec for mp4
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

        if post:
            self.output_file_frame_post = frame_name + "_post_"+ str(recording_id) + ".mp4"
            self.process2 = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(self.frame_width, self.frame_height), framerate = self.framerate)
                .output(self.output_file_frame_post, vcodec='libx264', pix_fmt='yuv420p') # Using h264 codec for mp4
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

    def record_frame(self, frame, coords):
        self.process.stdin.write(frame.tobytes())
        self.output_np_gaze.append(coords)
        self.timestamps.append(time.time())

    def record_frame_post(self, frame):
        # no need to check, this function shouldn't be called if post is false
        self.process2.stdin.write(frame.tobytes())

    def end_recording(self):
        # Clean up and finish writing to video
        self.process.stdin.close()
        self.process.wait()
        print(f'Recording Terminated.')
        print(f"Video saved to {self.output_file_frame}")

        # Finish writing gaze coordinate information
        np.save(self.output_file_gaze, np.array(self.output_np_gaze, dtype=object))
        print(f'Gaze coordinates saved to {self.output_file_gaze}')

        if self.record_timestamp:
            print('test')
            np.asarray(self.timestamps, dtype=np.float64)
            np.save(self.output_file_timestamps, self.timestamps)
            print(f'Timestamps saved to {self.output_file_timestamps}')

        if self.post:
            self.process2.stdin.close()
            self.process2.wait()
            print(f'Recording with Post-Processing Terminated.')
            print(f"Video saved to {self.output_file_frame_post}")

        print('Recording Ended Successfully!')
