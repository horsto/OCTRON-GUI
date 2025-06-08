#### Changelog

## vers. 0.0.4 
- Date 2025-05-08
- High level access to `YOLO_octron` and `YOLO_results` classes via, for example, `from octron import YOLO_results` 
- New notebook that explains the usage of these classes under "octron/notebooks/OCTRON_results_loading.ipynb"
- New `get_frame_indices` in `YOLO_results`: Helper to just return the valid frame indices for all track IDs.
- Enable `YOLO_results` to access results regardless of whether the original video was found or not. 
- Mask prediction results are now directly read from masks created in YOLO, instead of going through polygon-mask conversion steps. This is more efficient and less error prone. The Gaussian smoothing sigma parameter for prediction polygons (GUI and code) has been replaced with an `opening` (binary opening of masks) parameter. Morphological opening is (optionally) applied, which, similarly to the original smoothing sigma, can help to improve mask results. 
- Feature columns (eccentricity, area, ...) are now also interpolated with `interpolate=True` alongside position data
- During prediction frame and mask CIE LAB average values are extracted and saved in the .csv output. The experimenter thereby has access to color and brightness information for every frame and extracted mask after prediction completed. These values are new additions to the features columns, alongside eccentricity, area, etc. 
- Major update of ultralytics (8.3.152) that gets rid of an offest of mask vs. frame data introduced when the masks are scaled back to the original image size after prediction. See [PR 20957](https://github.com/ultralytics/ultralytics/pull/20957). 
- Created wheels for quick installation of py-av, sam2, and sam2-hq

## vers. 0.0.3
- Date: 2025-05-25
- Added buttons in annotation tab that allows users to skip to next / previous annotated frame. 
- Added `YOLO_results` class for unified access to OCTRON prediction results. Do `from octron import YOLO_results` to use this class. 
- Added skip frame functionality to analysis (prediction) of new videos in OCTRON which allows to analyze only a subset of the video frames in each video. 
- Added metadata export for each prediction that saves all parameters that the prediction has been run with.
- When loading prediction results via drag-n-drop on the OCTRON main window, the masks are now shown automatically and skipped frames are interpolated over. 
- MPS is now engaged on capable systems when training / predicting with YOLO.
- Retired YOLO Model X(tra large) from available model list for training since it is unnecessary.
- Shortened pruning method when creating training data. This needs to be checked for edge cases at some point, but runs much faster now. 

## vers. 0.0.2
- Date: 2025-04-15
- Initial working release.
- Implemented programmatic version update and tagging.