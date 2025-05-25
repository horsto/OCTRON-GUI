#### Changelog

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