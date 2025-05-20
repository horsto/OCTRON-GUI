from pathlib import Path
import zarr
import numpy as np
import pandas as pd
import warnings
# Plugins
from napari_pyav._reader import FastVideoReader


class YOLO_results:
    def __init__(self, results_dir, verbose=True, **kwargs):
        """
        
        
        Parameters
        ----------
        
        
        
        """
        # Ignore specific Zarr warning about .DS_Store
        # That happens on Mac ... might include more here.
        warnings.filterwarnings(
            "ignore",
            message="Object at .DS_Store is not recognized as a component of a Zarr hierarchy.",
            category=UserWarning,
            module="zarr.core.group"
        )
        
        # Process kwargs
        self.csv_header_lines = kwargs.get('csv_header_lines', 6)
        
        # Initialize some variables 
        self.verbose = verbose
        self.video, self.video_dict = None, None
        self.width, self.height, self.num_frames = None, None, None 
        self.csvs = None
        self.zarr, self.zarr_root = None, None
        self.track_ids = None
        self.labels = None
        self.track_id_label = None
        
        results_dir = Path(results_dir)
        assert results_dir.exists(), f"Path {results_dir.as_posix()} does not exist"
        self.results_dir = results_dir
        self.find_video()
        self.find_csv()
        self.find_zarr_root()
        _,  _, _ = self.get_track_ids_labels(csv_header_lines=self.csv_header_lines)
        
    def find_video(self):
        """
        Check if video is present in the second parent directory
        OCTRON saves results of analyzed mp4 files into a subdirectory /predictions/VIDEONAME/
        """
        from octron.sam2_octron.helpers.video_loader import probe_video
        results_dir = self.results_dir
        video = None
        video_dict = None
        for video_path in results_dir.parent.parent.glob('*.mp4'):
            if video_path.stem == '_'.join(results_dir.name.split('_')[:-1]):
                video = FastVideoReader(video_path, read_format='rgb24')
                video_dict = probe_video(video_path, verbose=self.verbose)
                self.height = video_dict['height']
                self.width = video_dict['width']
                self.num_frames = video_dict['num_frames']
                break
        if video is None and self.verbose:
            print(f"No video found for '{results_dir.name}'")
        self.video, self.video_dict = video, video_dict
            
    def find_csv(self):
        results_dir = self.results_dir
        csvs = list(results_dir.rglob('*.csv'))
        if not csvs and self.verbose:
            print(f"No tracking CSV files found in '{results_dir.name}'")
            self.csvs = None
        else:
            self.csvs = csvs
            if self.verbose:
                print(f"Found {len(csvs)} tracking CSV files in '{results_dir.name}'")
                      
    def find_zarr_root(self):
        """
        First find the zarr archive and then try to open the root group.
        
        """
        def _find_zarr():
            results_dir = self.results_dir
            zarrs = list(results_dir.rglob('*.zarr'))
            assert len(zarrs) == 1, f"Expected exactly one predictions zarr file, got {len(zarrs)}."
            zarr = zarrs[0]
            if not zarr and self.verbose:
                print(f"No tracking zarr found in '{results_dir.name}'")
                self.zarr = None
            else:
                self.zarr = zarr
                if self.verbose:
                    print(f"Found tracking zarr in '{results_dir.name}'")
                    
        _find_zarr()
        
        if self.zarr is not None:
            store = zarr.storage.LocalStore(self.zarr, read_only=False)
            root = zarr.open_group(store=store, mode='a')
            if self.verbose:
                print("Existing keys in zarr archive:", list(root.array_keys()))
            self.zarr_root = root
        else:
            raise ValueError("No zarr file found. Please run find_zarr() first.")   
        
    def define_colors(self, label_n=10, n_colors_submap=50):
        """
        Color handling: Recreate the colors here for the masks 
        This is a bit of a hack, but it works if the n_labels and n_colors_submap 
        match the original parameters used to create the colormap. 
        So, if we don't change these parameters, the colors will be the same, because 
        we are looking up object IDs from the original model classes
        """
        from octron.sam2_octron.helpers.sam2_colors import (create_label_colors, 
                                                            sample_maximally_different
                                                           )
        all_labels_submaps = create_label_colors(n_labels=label_n,
                                                 n_colors_submap=n_colors_submap,
                                                )
        indices_max_diff_labels    = sample_maximally_different(list(range(label_n)))
        indices_max_diff_subcolors = sample_maximally_different(list(range(n_colors_submap)))
        return all_labels_submaps, indices_max_diff_labels, indices_max_diff_subcolors
    
    def get_track_ids_labels(self, csv_header_lines=None): 
        """
        Compare track IDs that are present across csv files and the zarr archive.
        Track IDs are stored in the csvs (column "track_id"),
        and the zarr root as ".array_keys()".
        Corresponding label names are stored in the csvs (column "label").
        
        Returns
        -------
        track_ids : list
            Set of list of track IDs that are present in both the csvs and the zarr archive.
        labels : list
            Set of list of labels that are present in the csvs.
        track_id_label : dict
            Dictionary of track IDs and their corresponding labels.
            The keys are the track IDs and the values are the labels.
                
        """
        if csv_header_lines is not None:
            self.csv_header_lines = csv_header_lines
            
        def _track_ids_zarr():
            if self.zarr_root is not None:
                track_ids_zarr_keys = self.zarr_root.array_keys()
                # Keys are like "0_label", "1_label", etc.
                track_ids = [int(key.split('_')[0]) for key in track_ids_zarr_keys if key.split('_')[0].isdigit()]
                return sorted(list(set(track_ids)))
            else:
                if self.verbose:
                    print("Zarr root not found.")
                return []
        
        def _track_ids_labels_csv(header_lines):
            if self.csvs is not None:
                track_ids_from_csv = []
                labels_from_csv = []
                track_id_label_dict = {}
                for csv_file in self.csvs:
                    try:
                        df = pd.read_csv(csv_file, skiprows=header_lines, usecols=['track_id','label'])
                        if 'track_id' in df.columns:
                            assert set(df['track_id'].unique()) == set(df['track_id']), f"Duplicate track IDs found in {csv_file.name}"
                            track_ids_from_csv.extend(df['track_id'].unique().tolist())
                        elif self.verbose:
                            print(f"Column 'track_id' not found in {csv_file.name}")
                        if 'label' in df.columns:
                            assert set(df['label'].unique()) == set(df['label']), f"Duplicate labels found in {csv_file.name}"
                            labels_from_csv.extend(df['label'].unique().tolist())
                        elif self.verbose:
                            print(f"Column 'label' not found in {csv_file.name}")
                    except Exception as e:
                        if self.verbose:
                            print(f"Could not read track_ids and labels from {csv_file.name}: {e}")
                            
                # Write to a dictionary, sort the track_id key 
                for track_id, label in zip(track_ids_from_csv, labels_from_csv):
                    if track_id not in track_id_label_dict:
                        track_id_label_dict[track_id] = label
                    else:
                        raise ValueError(f"Duplicate track ID {track_id} found.")
                # Sort the dictionary by track_id
                track_id_label_dict = dict(sorted(track_id_label_dict.items()))
                return list(set(track_ids_from_csv)), list(set(labels_from_csv)), track_id_label_dict
            else:
                if self.verbose:
                    print("No CSV files found, cannot extract track IDs from CSVs.")
                return [], [], {}

        zarr_ids = _track_ids_zarr()
        csv_ids, csv_labels, track_id_label  = _track_ids_labels_csv(self.csv_header_lines)

        if not zarr_ids: 
            raise ValueError("No track IDs found in zarr archive.")
        if not csv_ids:
            raise ValueError("No track IDs found in CSV files.")
        if not csv_labels:
            raise ValueError("No labels found in CSV files.")
        assert set(zarr_ids) == set(csv_ids), f"Track IDs zarr and CSVs do not match: {set(zarr_ids) - set(csv_ids)}"
        self.track_ids = sorted(csv_ids)
        self.labels = sorted(csv_labels)
        self.track_id_label = track_id_label
        if self.verbose:
            print(f"Found {len(self.track_id_label)} unique track IDs in zarr and CSVs: {self.track_id_label}")

        return self.track_ids, self.labels, self.track_id_label
    
    
    def get_track_id_for_label(self, label):
        """
        Get the track ID for a given label.
        
        Parameters
        ----------
        label : str
            The label to search for.
            
        Returns
        -------
        track_id : int
            The track ID for the given label.
            
        """
        if self.track_id_label is None:
            raise ValueError("No track IDs found. Please run get_track_ids_labels() first.")
        
        track_id = self.find_class_by_name(self.track_id_label, label)
        if track_id is None:
            raise ValueError(f"Label '{label}' not found in track IDs.")
        # TODO:
        # Check what happens with multiple track IDs for the same label 
        
        return track_id
    
    
    def get_label_for_track_id(self, track_id): 
        """
        Get the label for a given track ID.
        
        Parameters
        ----------
        track_id : int
            The track ID to search for.
            
        Returns
        -------
        label : str
            The label for the given track ID.
            
        """
        if self.track_id_label is None:
            raise ValueError("No track IDs found. Please run get_track_ids_labels() first.")
        
        label = self.track_id_label.get(track_id, None)
        if label is None:
            raise ValueError(f"Track ID '{track_id}' not found in labels.")
        
        return label
    
    
    #### CORE DATA EXTRACTION ##############################################################
    
    def get_tracking_data(self):
        """
        Get the tracking data for all csvs in a dictionary
        of track_id -> tracking columns
        
        Returns
        -------
        tracking_data : dict
            Dictionary of track_id -> dict (label, data)
            Where data is a pandas dataframe with the tracking data.
            
        """
        if self.csvs is None:
            raise ValueError("No CSV files found, cannot extract tracking data.")
        
        tracking_data = {}
        for csv_file in self.csvs:
            try:
                df = pd.read_csv(csv_file, skiprows=self.csv_header_lines)
                track_id = df.iloc[0].track_id
                label = df.iloc[0].label
                if track_id not in tracking_data:
                    tracking_data[int(track_id)] = {
                        'label': label,
                        'data': df
                    }
                else:
                    raise ValueError(f"Duplicate track ID {track_id} found.")
            except Exception as e:
                if self.verbose:
                    print(f"Could not read tracking data from {csv_file.name}: {e}")
        return tracking_data
    
    
    def get_tracking_for_label(self, label):
        """
        Get the tracking data for only one label.
        
        Parameters
        ----------
        label : str
            The label to search for.
            
        Returns
        -------
        tracking_data : pandas.DataFrame
            The tracking data for the given label.
        frame_indices : numpy.ndarray
            The frame indices for which tracking data exist.
        """
        if self.track_id_label is None:
            raise ValueError("No track IDs found. Please run get_track_ids_labels() first.")
        track_id = self.get_track_id_for_label(label)
        tracking_data = self.get_tracking_data()
        if track_id not in tracking_data:
            raise ValueError(f"Track ID '{track_id}' not found in tracking data.")
        
        # Find out which indices have data
        frame_indices =  tracking_data[track_id]['data']['frame_idx'].values
        return tracking_data[track_id]['data'], frame_indices
    
    def get_mask_data(self):
        """
        Get the mask data for all track IDs in a dictionary
        of track_id -> mask data
        
        Returns
        -------
        mask_data : dict
            Dictionary of track_id -> mask data
            Where mask data is a numpy array with the mask data.
            
        """
        if self.zarr_root is None:
            raise ValueError("No zarr root found, cannot extract mask data.")
        assert self.track_ids is not None, "No track IDs found. Please run get_track_ids_labels() first."
        assert self.track_id_label is not None, "No track ID labels found. Please run get_track_ids_labels() first."
        
        mask_data = {}
        for track_id in self.track_ids:
            label = self.get_label_for_track_id(track_id)
            try:
                mask_key = f"{track_id}_masks"
                if mask_key in self.zarr_root.array_keys():
                    masks = self.zarr_root[mask_key]
                    num_frames = masks.shape[0]
                    height = masks.shape[1]
                    width = masks.shape[2]
                    if self.num_frames is not None: 
                        assert num_frames == self.num_frames, \
                            f"Number of frames in mask data ({num_frames}) does not match number of frames in video ({self.num_frames})."
                    if self.height is not None:
                        assert height == self.height, \
                            f"Height in mask data ({height}) does not match height in video ({self.height})."
                    if self.width is not None:
                        assert width == self.width, \
                            f"Width in mask data ({width}) does not match width in video ({self.width})."
                    mask_data[track_id] = {
                        'label': label,
                        'data' : masks,
                    }
                else:
                    if self.verbose:
                        print(f"Mask key '{mask_key}' not found in zarr archive.")  
            except Exception as e:
                if self.verbose:
                    print(f"Could not read mask data for track ID {track_id}: {e}")
        return mask_data
    
    def get_masks_for_label(self, label):
        """
        Get the mask data for only one label.
        
        Parameters
        ----------
        label : str
            The label to search for.
            
        Returns
        -------
        mask_data : zarr.Array
            The mask data for the given label.
        frame_indices : numpy.ndarray
            The frame indices for which data exist in the mask data.
        """
        if self.track_id_label is None:
            raise ValueError("No track IDs found. Please run get_track_ids_labels() first.")
        track_id = self.get_track_id_for_label(label)
        mask_data = self.get_mask_data()
        if track_id not in mask_data:
            raise ValueError(f"Track ID '{track_id}' not found in mask data.")
        
        # Find out which indices have data
        frame_indices = np.where(
            (mask_data[track_id]['data'][:,0,0] != -1)
        )[0]
        if len(frame_indices) == 0:
            raise ValueError(f"No valid frames found for track ID '{track_id}' in mask data.")
        return mask_data[track_id]['data'], frame_indices
    
    
    def get_masked_video_frames(self, label, frame_indices): 
        """
        Get the masked video frames for a given label and frame indices.
        
        Parameters
        ----------
        label : str
            The label to search for.
        frame_indices : numpy.ndarray
            The frame indices to search for.
            
        Returns
        -------
        masked_frames : list or numpy.ndarray
            The masked video frames for the given label and frame indices.
            If only one frame is requested, a numpy array is returned.
            If multiple frames are requested, a list of numpy arrays is returned, one for each frame.
            
        """
        if self.video is None:
            raise ValueError("No video found, cannot extract masked video frames.")
        if isinstance(frame_indices, int):
            frame_indices = [frame_indices]
        mask_data, _ = self.get_masks_for_label(label)
        masked_frames = []
        for frame_idx in frame_indices:
            frame = self.video[frame_idx]
            mask = mask_data[frame_idx]
            if frame.ndim == 3:
                mask = mask[:,:, np.newaxis]
            masked_frame = frame.copy()
            masked_frame = np.multiply(masked_frame, mask)
            masked_frames.append(masked_frame)
        
        if len(masked_frames) == 1:
            masked_frames = masked_frames[0]
        return masked_frames
    

    
    def __repr__(self) -> str:
        if self.num_frames is not None and self.width is not None and self.height is not None:
            return f"YOLO_results\n{self.results_dir}\n{self.num_frames} frames, {self.width}x{self.height}"
        else:
            return f"YOLO_results\n{self.results_dir}"
    def __str__(self) -> str:
        if self.num_frames is not None and self.width is not None and self.height is not None:
            return f"YOLO_results\n{self.results_dir}\n{self.num_frames} frames, {self.width}x{self.height}"
        else:
            return f"YOLO_results\n{self.results_dir}"
    
    #### OTHER HELPERS #######################################################################
    
    def find_class_by_name(self, classes, class_name):
            return (next((k for k, v in classes.items() if v == class_name), None))
