# Project Aria Gaze
Using gaze from Meta's aria glasses in for robot policy learning

For full official Aria documentation, see https://facebookresearch.github.io/projectaria_tools/docs/intro

# Requirements
- x64 Linux distributions of:
  - Fedora 36, 37, 38
  - Ubuntu focal (20.04 LTS) and jammy (22.04 LTS)
  - Windows (MSVC 2019/2022)
- Mac Intel or Mac ARM-based (M1) with MacOS 11 (Big Sur) or newer

# Installation Guide
We provide instructions for both running our project code AND a full installation of the entire original codebase from Facebook (Meta) Research. For full installation from source, see "Installation from Official Source" below this installation guide. 

1. Clone the github repository onto your local machine
```
git clone https://github.com/UCLA-Robot-Intelligence-Lab/aria-gaze
```

2. Create the anaconda environment:
```
conda env create -f environment.yml
```

3. Activate the environemnt and install all libraries
Make sure that you are in the right directory!
```
conda activate aria
pip install -r requirements.txt
```

To begin livestreaming on aria glasses, see [these instructions](streaming/instructions.md)

# Installation from Official Source

1. Create virtual environment (using either venv or conda, both have been tested and work)
```
conda create -n aria python=3.9.20
conda activate aria
```

2. Clone codebase (official aria codebase from Meta)
```
git clone https://github.com/facebookresearch/projectaria_tools.git -b 1.5.5
```
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
