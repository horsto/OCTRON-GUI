from pathlib import Path
import zarr

# Plugins
from napari_pyav._reader import FastVideoReader


class YOLO_results:
    def __init__(self, results_dir, verbose=True):
        """
        
        
        Parameters
        ----------
        
        
        
        """
        # Initiliaze some variables 
        self.verbose = verbose
        self.video, self.video_dict = None, None
        self.csvs = None
        self.zarr = None
        
        results_dir = Path(results_dir)
        assert results_dir.exists(), f"Path {results_dir.as_posix()} does not exist"
        self.results_dir = results_dir
        self.find_video()
        self.find_csv()
        self.find_zarr_root()
        
    
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
            assert len(zarrs) == 1, f"Expected exactly one predictions zarr file, got {len(zarrs)}"
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
    
    
    #### OTHER HELPERS #######################################################################
    
    def find_class_by_name(self, classes, class_name):
            return (next((k for k, v in classes.items() if v == class_name), None))
        