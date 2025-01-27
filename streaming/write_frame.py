
import cv2
import ffmpeg
import numpy as np
import time

class DataRecorder():
    def __init__(self, frame_name="gaze_data.mp4", coord_name="gaze_data.npy", framerate=10):
        self.output_file_frame = frame_name
        self.output_file_gaze = coord_name
        self.output_np_gaze = []
        self.framerate = int(framerate)

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

    def record_frame(self, frame, coords):
        self.process.stdin.write(frame.tobytes())
        self.output_np_gaze.append(coords)


    def end_recording(self):
        # Clean up and finish writing to video
        self.process.stdin.close()
        self.process.wait()
        print(f'Recording Terminated.')
        print(f"Video saved to {self.output_file_frame}")

        # Finish writing gaze coordinate information
        np.save(self.output_file_gaze, np.array(self.output_np_gaze, dtype=object))
        print(f'Gaze coordinates saved to {self.output_file_gaze}')