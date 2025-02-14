import os 
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import numpy as np
import zarr
import torch
from torchvision.transforms import Resize

def create_image_zarr(zip_path, 
                      num_frames, 
                      image_height,
                      image_width=None,
                      chunk_size=20,
                      ):
    '''
    Creates a zarr archive for storing and retrieving image data.
    This is meant for the resized images, resized to SAM2 predictor input size.
    
    Parameters
    ---------
    zip_path : pathlib.Path
        Path to the zarr archive. Must end in .zip
    num_frames : int
        Number of frames in the video
    image_height : int
        Height of the resized image (as in SAM2 input)
    image_width : int
        Width of the resized image (as in SAM2 input)
        If None, then image_width = image_height
    chunk_size : int, optional
        Size of the chunk to store in the zarr archive. 
        
    Returns
    -------
    image_zarr : zarr.core.Array
        The zarr array to store the image data in 
        
    '''
    assert image_height > 0, f'image_height must be > 0, not {image_height}'
    if image_width is None:
        image_width = image_height
    assert isinstance(zip_path, Path), f'path must be a pathlib.Path object, not {type(zip_path)}'
    assert zip_path.suffix == '.zip', f'path must be a .zip file, not {zip_path.suffix}'  
    
    if zip_path.exists():
        os.remove(zip_path)

    # Assuming local store on fast SSD, so no compression employed for now 
    store = zarr.storage.ZipStore(zip_path, mode='w')
    root = zarr.create_group(store=store)
    image_zarr = root.create_array(name='masks',
                                   shape=(num_frames, 3, image_height, image_width),  
                                   chunks=(chunk_size, 3, image_height, image_width), 
                                   fill_value=np.nan,
                                   dtype='float32'
                                   )
    return image_zarr

def load_image_zarr(zip_path, 
                    num_frames, 
                    image_height,
                    image_width=None,
                    chunk_size=20,
                    ):
    '''
    Loads an existing zarr archive for storing and retrieving image data,
    and checks if the stored array has the expected parameters.
    
    Parameters
    ---------
    zip_path : pathlib.Path
        Path to the zarr archive. Must end in .zip and exist.
    num_frames : int
        Expected number of frames in the video.
    image_height : int
        Expected height of the resized image (as in SAM2 input).
    image_width : int, optional
        Expected width of the resized image (as in SAM2 input). 
        If None, then image_width = image_height.
    chunk_size : int, optional
        Size of a chunk stored in the zarr archive.
        
    Returns
    -------
    image_zarr : zarr.core.Array
        The loaded zarr array storing the image data.
    status : bool
        True if the array was loaded successfully, False otherwise.   

    '''
    assert zip_path.exists(), f'Zip file {zip_path.as_posix()} does not exist.'
    assert image_height > 0, f'image_height must be > 0, not {image_height}'
    if image_width is None:
        image_width = image_height

    # Open the ZipStore in read mode and load the group.
    store = zarr.storage.ZipStore(zip_path, mode='a')
    root = zarr.open_group(store=store, mode='a')
    print("Existing keys in zarr archive:", list(root.array_keys()))
    # Attempt to load the array named 'masks'
    if 'masks' not in root:
        print(f"Array 'masks' not found in {zip_path.as_posix()}")
        return None, False
    else:
        image_zarr = root['masks']

    # Check shape: expected (num_frames, 3, image_height, image_width)
    expected_shape = (num_frames, 3, image_height, image_width)
    if image_zarr.shape != expected_shape:
        print(f"Shape mismatch: expected {expected_shape}, got {image_zarr.shape}")
        return None, False
    
    # Check chunks: expected (chunk_size, 3, image_height, image_width)
    expected_chunks = (chunk_size, 3, image_height, image_width)
    if image_zarr.chunks != expected_chunks:
        print(f"Chunk size mismatch: expected {expected_chunks}, got {image_zarr.chunks}")
        return None, False
    
    # Check dtype: expected float32
    if image_zarr.dtype != 'float32':
        print(f"dtype mismatch: expected float32, got {image_zarr.dtype}")
        return None, False
    
    return image_zarr, True


class OctoZarr:
    '''
    Flexible subclass of zarr array that allows for image data retrieval
    
    The idea here was to just replace the possibly very large 
    image dictionary directly with a zarr array. 
    I.e. instead of pre-loading all images into the dictionary, 
    just lazy load them when needed, and save them
    into zarr, so the second time they are accessed, the access is faster. 
    
    This should be optimized...  This is a lot (!) of back and forth (torch->numpy and back)
    
    
    '''
    def __init__(self, 
                 zarr_array, 
                 video_data,
                 running_buffer_size=50,
                 ):
        self.zarr_array = zarr_array
        self.saved_indices = []
        
        # Collect some basic info 
        num_frames, num_chs, image_height, image_width = zarr_array.shape
        assert image_height == image_width, f'Images in zarr store are not square'
        self.num_frames = num_frames
        self.num_chs = num_chs  
        self.image_size = image_height = image_width
        # The original implementation uses a fixed mean and std 
        img_mean = (0.485, 0.456, 0.406)
        img_std  = (0.229, 0.224, 0.225)
        self.img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        
        # Initialize resizing function
        self._resize_img = Resize(size=(self.image_size))

        # Store the napari data layer   
        self.video_data = video_data
        self.cached_indices = np.full((running_buffer_size), np.nan)
        self.cached_images  = torch.empty(running_buffer_size, self.num_chs, self.image_size, self.image_size)
        self.cur_cache_idx = 0 # Keep track of where you are in the cache currently

    @property
    def indices_in_store(self):
        return self.saved_indices        

    def _save_to_zarr(self, batch, indices):
        ''' 
        Save a batch of images to  zarr array at index position indices
        '''
        assert len(indices), 'No indices provided'
        assert len(batch) == len(indices), 'Batch and indices should have the same length'
        
        if len(batch) == 1:
            batch = batch[0]
        if len(indices) == 1:
            indices_ = indices[0]
        else:
            indices_ = indices
        self.zarr_array[indices_,:,:,:] = batch.numpy()    
        self.saved_indices.extend(indices)   
         
    @torch.inference_mode()
    def _fetch_one(self, idx):
        img = self.video_data[idx]
        img = self._resize_img(torch.from_numpy(img).permute(2,0,1)).float()
        img /= 255.  
        img -= self.img_mean
        img /= self.img_std     
        # Cache 
        self.cached_indices[self.cur_cache_idx] = idx
        self.cached_images[self.cur_cache_idx] = img
        self.cur_cache_idx += 1
        if self.cur_cache_idx == len(self.cached_indices):
            self.cur_cache_idx = 0  
        return img   
    
    @torch.inference_mode()
    def _fetch_many(self, indices):
        imgs = self.video_data[indices]
        imgs = self._resize_img(torch.from_numpy(imgs).permute(0,3,1,2)).float()
        imgs /= 255.  
        imgs -= self.img_mean
        imgs /= self.img_std
        # Cache
        for idx, img in zip(indices, imgs):
            self.cached_indices[self.cur_cache_idx] = idx
            self.cached_images[self.cur_cache_idx] = img
            self.cur_cache_idx += 1
            if self.cur_cache_idx == len(self.cached_indices):
                self.cur_cache_idx = 0  
        return imgs
    
    def fetch(self, indices):   
        
        '''
        Check if the images are already in the zarr store.
        
        The logic is the following: 
        - Enable "quick" loading of single indices without saving them into zarr array. This would 
          just slow things down. 
        - Enable slightly slower loading from and saving to zarr for batches of images
        
        Generally for multiple images (batches):
        For those images that are not in the store, prefetch them from the napari data layer, then
        - Resize the images
        - Normalize the images
        - Save the images to the zarr array
        Combine those with images loaded from zarr store 
        - Return the combined images as torch tensor
        
        '''
        min_idx = np.min(indices)
        
        # Initialize empty torch arrach of length indices
        imgs_torch = torch.empty(len(indices), self.num_chs, self.image_size, self.image_size)
        
        # First check whether the indices are in the cache
        # If they are, return them immediately
        cached_idx = np.where(np.isin(self.cached_indices, indices))[0]
        if len(cached_idx):
            #print(f'Cached at indices {self.cached_indices[cached_idx]}')
            imgs_cached = self.cached_images[cached_idx]
            imgs_torch[np.where(np.isin(indices, self.cached_indices))[0]] = imgs_cached
            
        # Cover cases for which there are indices left (images that are not in the rolling cache)
        # Subtract the cached indices from the indices
        indices = np.setdiff1d(indices, self.cached_indices)
        if len(indices) == 1:
            # Single image
            idx = indices[0]
            if idx in self.saved_indices:
                img = torch.from_numpy(self.zarr_array[idx])
            else:
                img = self._fetch_one(idx=idx)
                self._save_to_zarr([img], [idx])
            imgs_torch[idx-min_idx] = img
            
        elif len(indices) > 1:
            # Create indices
            not_in_store = np.array([idx for idx in indices if idx not in self.saved_indices]).astype(int)
            in_store = np.array([idx for idx in indices if idx in self.saved_indices]).astype(int)
            zeroed_not_in_store = not_in_store - min_idx # for writing into `imgs_torch`
            zeroed_in_store = in_store - min_idx # for writing into `imgs_torch`
            
            if len(not_in_store):
                imgs = self._fetch_many(indices=not_in_store)
                imgs_torch[np.where(np.isin(indices, not_in_store))[0]] = imgs
                # Save this batch to zarr 
                self._save_to_zarr(imgs, not_in_store)
            if len(in_store):
                #print(f'Found in store (multiple): {in_store}')
                imgs_in_store = torch.from_numpy(self.zarr_array[in_store]).squeeze()
                imgs_torch[np.where(np.isin(indices, in_store))[0]] = imgs_in_store    

        return imgs_torch.squeeze()
    
    def __getitem__(self, frame_idx):
        '''
        Normal "get" function 

        '''
        if isinstance(frame_idx, slice):
            indices = np.arange(frame_idx.start, frame_idx.stop, frame_idx.step)
        elif isinstance(frame_idx, list):
            indices = np.array(frame_idx)
        elif isinstance(frame_idx, int):
            indices = [frame_idx]   
        else:
            raise ValueError(f'frame_idx should be int, list or slice, got {type(frame_idx)}')

        images = self.fetch(indices)
        return images
        
    def __repr__(self):
            return repr(self.zarr_array)
