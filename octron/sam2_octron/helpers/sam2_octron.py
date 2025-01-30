
from collections import OrderedDict
import numpy as np
import torch
from torchvision.transforms import Resize

from tqdm import tqdm
import napari
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import concat_points




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
                 napari_data,
                 running_buffer_size=150,
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
        self.napari_data = napari_data
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
        self.zarr_array[indices,:,:,:] = batch.numpy()    
        self.saved_indices.extend(indices)   
         
    @torch.inference_mode()
    def _fetch_one(self, idx):
        img = self.napari_data[idx]
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
        imgs = self.napari_data[indices]
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
        # Subtract the cached indices from the indices
        indices = np.setdiff1d(indices, self.cached_indices)
        # Cover cases for which there are indices left (images that are not in the rolling cache)
        if len(indices) == 1:
            # Single image
            idx = indices[0]
            if idx in self.saved_indices:
                #print('Found saved single image')
                img = torch.from_numpy(self.zarr_array[idx])
            else:
                img = self._fetch_one(idx=idx)
                # For now do not safe single images to zarr 
                # The intuition is that this would just slow things down
                # ... rather focus on saving batches of images
                # self._save_to_zarr(imgs_torch, idx)  
            imgs_torch[idx-min_idx] = img
        elif len(indices) > 1:
            # Create indices
            not_in_store = np.array([idx for idx in indices if idx not in self.saved_indices]).astype(int)
            in_store = np.array([idx for idx in indices if idx in self.saved_indices]).astype(int)
            zeroed_not_in_store = not_in_store - min_idx # for writing into `imgs_torch`
            zeroed_in_store = in_store - min_idx # for writing into `imgs_torch`
            
            if len(not_in_store):
                imgs = self._fetch_many(indices=not_in_store)
                imgs_torch[zeroed_not_in_store] = imgs
                # Save this batch to zarr 
                self._save_to_zarr(imgs, not_in_store)
            if len(in_store):
                #print('Found saved multiple images')
                imgs_in_store = torch.from_numpy(self.zarr_array[in_store]).squeeze()
                imgs_torch[zeroed_in_store] = imgs_in_store    

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



class SAM2_octron(SAM2VideoPredictor):
    '''
    Subclass of SAM2VideoPredictor that adds some additional functionality for OCTRON.
    '''
    def __init__(
        self,
        **kwargs,
    ):
        
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        
        super().__init__(**kwargs)
       
        
        print('\n\nLoaded SAM2VideoPredictor OCTRON')
        
    @torch.inference_mode()
    def init_state(
        self,
        napari_viewer,
        video_layer_idx,
        zarr_store
    ):
        '''
        Goal 
        ----    
        Feed in Napari video layer as is 
        - Calculate Mean and Std of the video layer
        - ...
        
        
        napari_data 
        
        
        '''
        compute_device = self.device  
        assert isinstance(napari_viewer, napari.Viewer), f"viewer should be a napari viewer, got {type(napari_viewer)}"
        self.viewer = napari_viewer
        
        
        napari_data = self.viewer.layers[video_layer_idx].data 
        assert len(napari_data.shape) == 4, f"video data should have shape (num_frames, H, W, 3), got {napari_data.shape}"
        assert napari_data.shape[3] == 3, f"video data should be RGB and have shape (num_frames, H, W, 3), got {napari_data.shape}"


        """Initialize an inference state."""
        inference_state = {}
        self.inference_state = inference_state 
        
        # Zarr store for the image data
        # zarr_chunk_size = zarr_store.chunks[0]
        # Replace the zarr array with the custom subclass
        self.images = OctoZarr(zarr_store, napari_data) 
        # Store the image data zarr in the inference state
        inference_state["images"] = self.images
        
        num_frames, video_height, video_width, _ = napari_data.shape
        inference_state["num_frames"] = num_frames 
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] =  video_height
        inference_state["video_width"]  =  video_width 
        
        inference_state["device"] = compute_device
        inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        print('Initialized SAM2 model')
        
        self.video_data = napari_data                               
                            
    
    def _run_single_frame_inference(
        self,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        
        
        '''
        This function is called both when new points / masks are added,
        as well as for (batched) video prediction.
        
        
        Run tracking on a single frame based on current inputs and previous memory.
        '''


        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(self.inference_state, frame_idx, batch_size)

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=self.inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = self.inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # # potentially fill holes in the predicted masks
        # if self.fill_hole_area > 0:
        #     pred_masks_gpu = fill_holes_in_mask_scores(
        #         pred_masks_gpu, self.fill_hole_area
        #     )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(self.inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu



    @torch.inference_mode()
    def propagate_in_video(
        self,
        start_frame_idx,
        max_frame_num_to_track,
        reverse=False,
        ):
        
        # Fetch and cache images before starting the tracking
        _ = self.images[slice(start_frame_idx,start_frame_idx+max_frame_num_to_track)]
        
        self.propagate_in_video_preflight(self.inference_state)

        obj_ids = self.inference_state["obj_ids"]
        num_frames = self.inference_state["num_frames"]
        batch_size = self._get_obj_num(self.inference_state)

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t
                for obj_output_dict in self.inference_state["output_dict_per_obj"].values()
                for t in obj_output_dict["cond_frame_outputs"]
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
                    
        try:
            for frame_idx in tqdm(processing_order, desc="Predicting"):
                pred_masks_per_obj = [None] * batch_size
                for obj_idx in range(batch_size):
                    obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
                    # We skip those frames already in consolidated outputs (these are frames
                    # that received input clicks or mask). Note that we cannot directly run
                    # batched forward on them via `_run_single_frame_inference` because the
                    # number of clicks on each object might be different.
                    if frame_idx in obj_output_dict["cond_frame_outputs"]:
                        storage_key = "cond_frame_outputs"
                        current_out = obj_output_dict[storage_key][frame_idx]
                        device = self.inference_state["device"]
                        pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                        
                        # TODO: Reimplement this function 
                        # if self.clear_non_cond_mem_around_input:
                            # # clear non-conditioning memory of the surrounding frames
                            # self._clear_obj_non_cond_mem_around_input(
                            #     self.inference_state, frame_idx, obj_idx
                            # )
                    else:
                        storage_key = "non_cond_frame_outputs"
                        current_out, pred_masks = self._run_single_frame_inference(
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=True,
                        )
                        
                        obj_output_dict[storage_key][frame_idx] = current_out
                        
                        #Clear all non conditioned output frames that are older than 16 frames
                        #https://github.com/facebookresearch/sam2/issues/196#issuecomment-2286352777
                        oldest_allowed_idx = frame_idx - 16
                        all_frame_idxs = obj_output_dict[storage_key].keys()
                        old_frame_idxs = [idx for idx in all_frame_idxs if idx < oldest_allowed_idx]
                        for old_idx in old_frame_idxs:
                            obj_output_dict[storage_key].pop(old_idx)
                            for objid in self.inference_state['output_dict_per_obj'].keys():
                                if old_idx in self.inference_state['output_dict_per_obj'][objid][storage_key]:
                                    self.inference_state['output_dict_per_obj'][objid][storage_key].pop(old_idx)
                                
                    self.inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                        "reverse": reverse
                    }
                    pred_masks_per_obj[obj_idx] = pred_masks

                # Resize the output mask to the original video resolution (we directly use
                # the mask scores on GPU for output to avoid any CPU conversion in between)
                if len(pred_masks_per_obj) > 1:
                    all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
                else:
                    all_pred_masks = pred_masks_per_obj[0]
                _, video_res_masks = self._get_orig_video_res_output(
                    self.inference_state, all_pred_masks
                )
                yield frame_idx, obj_ids, video_res_masks
        except Exception as e:
            print(e)
            pass

            
    @torch.inference_mode()
    def reset_state(self):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results()
        # Remove all object ids
        self.inference_state["obj_id_to_idx"].clear()
        self.inference_state["obj_idx_to_id"].clear()
        self.inference_state["obj_ids"].clear()
        self.inference_state["point_inputs_per_obj"].clear()
        self.inference_state["mask_inputs_per_obj"].clear()
        self.inference_state["output_dict_per_obj"].clear()
        self.inference_state["temp_output_dict_per_obj"].clear()
        self.inference_state["frames_tracked_per_obj"].clear()


    def _reset_tracking_results(self):
        """Reset all tracking inputs and results across the videos."""
        for v in self.inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in self.inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in self.inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in self.inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in self.inference_state["frames_tracked_per_obj"].values():
            v.clear()
            
            
    @torch.inference_mode()
    def remove_object(self, obj_id, strict=False, need_output=True):
        """
        Remove an object id from the tracking state. If strict is True, we check whether
        the object id actually exists and raise an error if it doesn't exist.
        """
        old_obj_idx_to_rm = self.inference_state["obj_id_to_idx"].get(obj_id, None)
        updated_frames = []
        # Check whether this object_id to remove actually exists and possibly raise an error.
        if old_obj_idx_to_rm is None:
            if not strict:
                return self.inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {self.inference_state['obj_ids']}."
            )

        # If this is the only remaining object id, we simply reset the state.
        if len(self.inference_state["obj_id_to_idx"]) == 1:
            self.reset_state()
            return self.inference_state["obj_ids"], updated_frames

        # There are still remaining objects after removing this object id. In this case,
        # we need to delete the object storage from inference state tensors.
        # Step 0: clear the input on those frames where this object id has point or mask input
        # (note that this step is required as it might downgrade conditioning frames to
        # non-conditioning ones)
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            self.inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            self.inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_prompts_in_frame(
                self.inference_state, frame_idx, obj_id, need_output=False
            )

        # Step 1: Update the object id mapping (note that it must be done after Step 0,
        # since Step 0 still requires the old object id mappings in self.inference_state)
        old_obj_ids = self.inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        self.inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        self.inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        self.inference_state["obj_ids"] = new_obj_ids

        # Step 2: For per-object tensor storage, we shift their obj_idx in the dict keys.
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(self.inference_state["point_inputs_per_obj"])
        _map_keys(self.inference_state["mask_inputs_per_obj"])
        _map_keys(self.inference_state["output_dict_per_obj"])
        _map_keys(self.inference_state["temp_output_dict_per_obj"])
        _map_keys(self.inference_state["frames_tracked_per_obj"])

        # Step 3: Further collect the outputs on those frames in `obj_input_frames_inds`, which
        # could show an updated mask for objects previously occluded by the object being removed
        if need_output:
            temp_output_dict_per_obj = self.inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    self.inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    self.inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return self.inference_state["obj_ids"], updated_frames
    
    
    ######## ADDING NEW POINTS AND MASKS ################################################################
    #####################################################################################################
    
    
    
    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(self.inference_state, obj_id)
        point_inputs_per_frame = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = self.inference_state["video_height"]
            video_W = self.inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(self.inference_state["device"])
        labels = labels.to(self.inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        obj_frames_tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = self.inference_state["device"]
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = self.inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            self.inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            self.inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

   
    @torch.inference_mode()
    def add_new_mask(
        self,
        frame_idx,
        obj_id,
        mask,
    ):
        """Add new mask to a frame."""
        obj_idx = self._obj_id_to_idx(self.self.inference_state, obj_id)
        point_inputs_per_frame = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(self.inference_state["device"])

        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        obj_frames_tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = self.inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            self.inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            self.inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks
