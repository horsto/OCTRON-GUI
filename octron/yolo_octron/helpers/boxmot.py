import numpy as np
from pathlib import Path
import yaml

class BoxMOTWrapper:
    """
    
    Wrapper for BoxMOT trackers to integrate with YOLO_octron
    
    """
    
    def __init__(self, tracker_type='bytetrack', config_path=None):
        """
        Initialize BoxMOT tracker
        
        Parameters
        ----------
        tracker_type : str
            Type of tracker ('bytetrack' or 'botsort')
        config_path : str or Path
            Path to config file
        """
        try:
            from boxmot import create_tracker
        except ImportError:
            raise ImportError("BoxMOT is required. Install with: pip install 'boxmot[YOLO]'")
            
        self.tracker_type = tracker_type.lower()
        assert self.tracker_type in ['bytetrack', 
                                     'botsort',
                                     'ocsort', 
                                     'deepocsort',
                                     'strongsort',
                                     'boosttrack'
                                     ], f"Unsupported tracker type: {tracker_type}"
        
        # Load configuration
        if config_path is None:
            raise ValueError("config_path must be provided")
        config_path = Path(config_path)
        with config_path.open('r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize tracker
        self.tracker = create_tracker(
            tracker_type=self.tracker_type,
            tracker_config=self.config,
            reid_weights=None,
            device='cpu',
            half=False
        )
        
    def update(self, dets, cls_ids, frame):
        """
        Update tracker with new detections
        
        Parameters
        ----------
        dets : ndarray
            Detections in format [x1, y1, x2, y2, conf]
        cls_ids : ndarray
            Class IDs for each detection
        frame : ndarray
            Current video frame
            
        Returns
        -------
        ndarray
            Tracked objects in format [x1, y1, x2, y2, track_id, class_id, conf, detection_index]
            The last column contains the original detection index for mask association
        """
        # Format detections for BoxMOT
        if len(dets) == 0:
            return np.empty((0, 8))  # Return empty array with correct shape
            
        # BoxMOT expects detections in format [x1, y1, x2, y2, conf, cls_id]
        boxmot_dets = np.concatenate([
            dets[:, :4],  # boxes
            dets[:, 4:5],  # confidence scores
            cls_ids.reshape(-1, 1)  # class IDs
        ], axis=1)
        
        # Update tracker
        outputs = self.tracker.update(boxmot_dets, frame)
        return outputs