# Project Aria Gaze
Using gaze from Meta's aria glasses in for robot policy learning

This project is a work in progress, the installation guide below is solely for the livestreaming functionality of the glasses on a Linux Ubuntu system. The installation guide will be updated later on in the project when we have our own code.

For full official Aria documentation, see https://facebookresearch.github.io/projectaria_tools/docs/intro

# Installation Guide

1. Create virtual environment (using either venv or conda, both have been tested and work)
```
conda create -n aria python=3.9.20
conda activate aria
```

2. Clone codebase (official aria codebase from Meta)
`git clone https://github.com/facebookresearch/projectaria_tools.git -b 1.5.5`
* the full codebase is only required for visualization, the actual livestreaming only requires common.py, test.py, and visualizer.py from the sdk files

4. Install required python dependencies
```
python3 -m pip install --upgrade pip
python3 -m pip install projectaria-tools'[all]'
python3 -m pip install projectaria_client_sdk --no-cache-dir # Aria client SDK install (for livestreaming)
```

5. Run aria-doctor to check for issues
```
aria-doctor
```

7. Pair glasses to computer
```
aria auth pair
```

After running this command in terminal, go to the Aria app on your phone and approve the pairing

7. Extract code for livestreaming
```
python -m aria.extract_sdk_samples --output ~
cd ~/projectaria_client_sdk_samples
python3 -m pip install -r requirements.txt
```

8. Run streaming file
```
aria streaming start --interface usb --use-ephemeral-certs
python -m streaming_subscribe
```
