# Livestreaming Gaze + RGB with Aria Glasses

Official documentation currently only contains code for aria livestreaming for RGB images (from the glasses), but we have created a fully functional workaround for livestreaming the gaze estimations as well. We still utilize only libraries that are defined within the scope of project Aria, as well as the gaze estimation model referenced in the official documentation (see below for more info).

The current streaming code contains live gaze estimation synced with the RGB camera. The [gaze_model folder](gaze_model) contains the gaze estimation model (see more info [here](https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/mps_eye_gaze)) for model's weights and configurations that are loaded in streaming subscribe. The inference speed for the model is approximately 0.003s (averaged).

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

