from pathlib import Path
from typing import Union, Sequence, Callable, List, Optional
from napari.types import LayerData
from napari.utils.notifications import (
    show_error,
)
# Define some types
PathLike = str
PathOrPaths = Union[PathLike, Sequence[PathLike]]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]]

import warnings
warnings.simplefilter("once")


def octron_reader(path: "PathOrPaths") -> Optional["ReaderFunction"]:
    """
    OCTRON napari reader.
    Accepts OCTRON project folders.
    
    Parameters
    ----------
    path : str or list of str
        Path to a file or folder.
    
    Returns
    -------
    function : Callable
        Function to read the file or folder.
        
    """
    
    path = Path(path)
    if path.is_dir() and path.exists():
        return read_octron_folder
        
    if path.is_file() and path.exists():
        return read_octron_file

def read_octron_file(path: "PathOrPaths") -> List["LayerData"]:
    """
    Single file reads that are dropped in the main window are not supported.
    """
    show_error(
        f"Single file drops to main window are not supported"
    )
    return [(None,)]

def read_octron_folder(path: "PathOrPaths") -> List["LayerData"]:
    path = Path(path)
    # Check what kind of folder you are dealing with.
    # There are three options:
    # A. Octron project folder
    # B. Octron video (annotation) folder
    # C. Octron prediction (results) folder
    # D. Video folder to transcribe to mp4
    
    # Case A 
    
    
    
    
    
    
    # Case C 
    # Check if the folder has .csv files AND a predictions.zarr 
    csvs = list(path.glob("*.csv"))
    prediction_zarr = list(path.glob("predictions.zarr"))
    if csvs and prediction_zarr:
        print(
            f"🐙 Detected OCTRON prediction folder: {path}"
        )
        # Load predictions
        from octron.yolo_octron.yolo_octron import YOLO_octron
        yolo_octron = YOLO_octron()
        yolo_octron.show_predictions(
            save_dir = path,
            sigma_tracking_pos = 2, # Fixed for now 
        )
        return [(None,)]
    
    
    # Case D 
    # Check if the folder has any kind of video or mj2 files 
    video_formats = [".avi", ".mov", ".mj2", ".mjpeg", ".wmv", ".mp4", ".mkv"]
    
    # Find all video files in the folder
    video_files = []
    for fmt in video_formats:
        video_files.extend(list(path.glob(f"*{fmt}")))
    
    # If we found video files, offer to transcode them
    if video_files:
        print(f"🎬 Found {len(video_files)} video files in {path}")
        
        # Create a dialog for transcoding options
        from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                   QCheckBox, QSpinBox, QPushButton, QListWidget,
                                   )
        from qtpy.QtCore import QSize
        
        dialog = QDialog()
        dialog.setWindowTitle("Transcode Videos")
        layout = QVBoxLayout()
        
        # Add description
        layout.addWidget(QLabel(f"Found {len(video_files)} videos to transcode to MP4:"))
        
        # Add file list
        file_list = QListWidget()
        for video in video_files:
            file_list.addItem(video.name)
        layout.addWidget(file_list)
        
        # Add options
        options_layout = QHBoxLayout()
        
        # Create subfolder option
        subfolder_check = QCheckBox("Create subfolder")
        subfolder_check.setChecked(True)
        options_layout.addWidget(subfolder_check)
        
        # CRF value option
        crf_layout = QHBoxLayout()
        crf_layout.addWidget(QLabel("CRF (lower is better):"))
        crf_spin = QSpinBox()
        crf_spin.setRange(0, 51)
        crf_spin.setValue(23)  # Default CRF value
        crf_spin.setSingleStep(1)
        crf_spin.setMinimumSize(QSize(60, 25))
        crf_spin.setMaximumSize(QSize(60, 25))
        crf_layout.addWidget(crf_spin)
        options_layout.addLayout(crf_layout)
        
        layout.addLayout(options_layout)
        
        # Add buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        # Connect buttons
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        # Show dialog and wait for user input
        if dialog.exec_():
            # User clicked OK, process videos
            import subprocess
            import os
            from concurrent.futures import ThreadPoolExecutor
            from tqdm import tqdm
            
            # Get options
            create_subfolder = subfolder_check.isChecked()
            crf_value = crf_spin.value()
            
            # Create output folder if needed
            if create_subfolder:
                output_folder = path / "mp4_transcoded"
                output_folder.mkdir(exist_ok=True)
            else:
                output_folder = path
                
            print(f"🔄 Transcoding {len(video_files)} videos to MP4 (CRF: {crf_value})...")
            
            # Function to transcode a single video
            def transcode_video(video_path):
                output_path = output_folder / f"{video_path.stem}.mp4"
                cmd = [
                    "ffmpeg", "-i", str(video_path), 
                    "-c:v", "libx264", "-preset", "superfast", 
                    "-crf", str(crf_value),
                    "-c:a", "aac", "-b:a", "128k",  # Audio settings
                    str(output_path),
                    "-y"  # Overwrite output if it exists
                ]
                
                try:
                    subprocess.run(
                        cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        check=True
                    )
                    return (True, video_path, output_path)
                except subprocess.CalledProcessError as e:
                    return (False, video_path, str(e))
            
            # Process videos in parallel
            results = []
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                for result in tqdm(
                    executor.map(transcode_video, video_files),
                    total=len(video_files),
                    desc="Transcoding videos"
                ):
                    results.append(result)
            
            # Report results
            successful = sum(1 for r in results if r[0])
            print(f"✅ Successfully transcoded {successful}/{len(video_files)} videos")
            
            # Show any errors
            for success, video_path, error in results:
                if not success:
                    print(f"❌ Failed to transcode {video_path.name}: {error}")
        
        return [(None,)]
    
    return [(None,)]


