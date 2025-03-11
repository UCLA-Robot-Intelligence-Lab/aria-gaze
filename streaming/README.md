# Streaming & Recording Gaze + RGB with Aria Glasses

Official documentation currently only contains code for aria livestreaming for RGB images (from the glasses), but we have created a fully functional workaround for livestreaming the gaze estimations as well. We still utilize only libraries that are defined within the scope of project Aria, as well as the gaze estimation model referenced in the official documentation (see below for more info).

The current streaming code contains live gaze estimation synced with the RGB camera. The [gaze_model folder](gaze_model) contains the gaze estimation model (see more info [here](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/mps_eye_gaze)) for model's weights and configurations that are loaded in streaming subscribe. The inference speed for the model is approximately 0.003s (averaged).

# Streaming

### Running streaming subscription file

1. Start streaming on your aria glasses with the following command

Aria allows for streaming via both USB and wifi. However, to begin streaming on either, you must be connected to your computer via USB when running the following commands.

To stream over USB (i.e. the glasses will remain connected via USB when moving around), run the below command
```
aria streaming start --interface usb --use-ephemeral-certs
```

If you wish to stream over wifi (no USB), use the folowing command to generate a certificate:
```
aria streaming start --interface wifi --device-ip <glasses IP address>
```
Replace the above glasses IP address with your IP address, found in the Aria app on your phone. Make sure your glasses are connected and that they appear in your dashboard. Your IP address is found under Wi-Fi.

2. Subscribe to stream using the command below

Run the python file for livestreaming using:
```
python -m streaming_subscribe --device-ip <glasses IP address>
```
Replace the above glasses IP address with your IP address.

3. To close the live images, click the opencv window that pops up with the livestream and click q (you can click on either the RGB or SLAM streams)
* Note, this will NOT shut down the stream, this will only close the viewer

4. To shut down the stream entirely on the glasses, run the following command
```
aria streaming stop
```
You can verify that the streaming has stopped via your Aria app on phone.

# Recording

Similar to streaming, make sure that your Aria glasses are on. Then, we have a few different options for streaming:

1. The [record_aria](record_aria.py) python file allows you to stream AND also record the footage from the aria glasses. This file does NOT stream or record the realsense robot camera. It will record the visual footage into an mp4 file and the gaze point into a .npy file.

```
python record_aria.py
```

2. The [record_aria_post](record_aria_post.py) python file allows you to stream AND also record the footage from the aria glasses WITH the gaze point annotated AND coordinates displayed on screen. It will still record the visual footage into an mp4 file and the gaze point into a .npy file.

```
python record_aria_post.py
```

3. The [record_realsense](record_realsense.py) python file allows you stream on BOTH the aria and realsense robot cameras simultaneously, as well as record the realsense footage (only), which is saved as an mp4 files, as well as the gaze point from the realsense's point of view, saved once again as a .npy file.

```
python record_realsense.py
```
