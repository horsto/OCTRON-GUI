# OCTRON



## Installation 

Follow these steps: 

1. Make sure ffmpeg is installed on the system
    ![FFmpeg Test](pics/ffmpeg_test.png)
    - If this output fails for some reason, make sure you install ffmpeg first:
        - [step by step guide for windows](ffmpeg_windows.md)
        - on MacOS you can use [homebrew](https://formulae.brew.sh/formula/ffmpeg) and `brew install ffmpeg`
        - Linux users: You know what to do

2. Download miniconda. Open your web browser and go to the official Miniconda download page: [Miniconda Download](https://docs.conda.io/en/latest/miniconda.html). Download and execute the installer for your operating system (Windows, macOS, or Linux). Then restart your terminal.

3. Create a new Conda environment called "octron" with Python version 3.11:
    ```sh
    conda create --name octron python=3.11 -y
    ```

4. Activate the new environment:
    ```sh
    conda activate octron
    ```
5. Clone this repository and browse to the folder that you cloned it to 
6. Install OCTRON:
    ```sh
    pip install -e .
    ```
    This installs the package and all the required libraries in "edit" mode. That means that if you change the code, at next execution of the plugin, it will run the newest (updated) version.
7. Check the accessibility of GPU resources on your computer:
    ```sh
    python test_gpu.py
    ```

## Usage
1. Activate the new environment:
    ```sh
    conda activate octron
    ```
2. Open Napari
    ```sh
    napari
    ```
3. Within Napari, OCTRON should appear under the Plugins Menu. Click it and *have fun*!

More instructions to follow ... stay tuned! 

![Octron main GUI](pics/octron_main_gui.png)

---

## Attributions
- Interface button and icon images were created by user [Arkinasi](https://thenounproject.com/browse/collection-icon/marketing-agency-239829/) from Noun Project (CC BY 3.0)
