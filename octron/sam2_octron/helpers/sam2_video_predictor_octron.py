from collections import OrderedDict
import numpy as np
import torch
from torchvision.transforms import Resize
from sam2.sam2_video_predictor import SAM2VideoPredictor
















class SAM2VideoPredictor_octron(SAM2VideoPredictor):
    '''
    Subclass of SAM2VideoPredictor that adds some additional functionality for OCTRON.
    '''
    def __init__(
        self,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        
        
        print('\n\nLoaded SAM2VideoPredictor OCTRON')
        
    @torch.inference_mode()
    def init_state(
        self,
        napari_data,
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
         
        assert len(napari_data.shape) == 4, f"video data should have shape (num_frames, H, W, 3), got {napari_data.shape}"
        assert napari_data.shape[3] == 3, f"video data should be RGB and have shape (num_frames, H, W, 3), got {napari_data.shape}"
        # Generic torch resize transformation
        self._resize_img = Resize(
                        size=(self.image_size) # This is 1024x1024 for the l model
                    )
        
        """Initialize an inference state."""
        inference_state = {}
        num_frames, video_height, video_width, _ = napari_data.shape
        inference_state["num_frames"] = num_frames 
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] =  video_height
        inference_state["video_width"]  =  video_width
        
        
        # Estimate mean and std of the video data on a subset of all frames 
        #self.mean, self.std = self._get_mean_std(napari_data, num_frames, max_num_frames=75)
        # The original implementation uses a fixed mean and std 
        img_mean = (0.485, 0.456, 0.406)
        img_std  = (0.229, 0.224, 0.225)
        self.img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        # Load first frame into inference state
        first_image = torch.from_numpy(napari_data[0]).permute(2, 0, 1)[torch.newaxis]
        first_image = self._resize_img(first_image)
        first_image = first_image.float() / 255.0
        # # #  # normalize by mean and std
        first_image -= self.img_mean
        first_image /= self.img_std
        
        inference_state["images"] = first_image   
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
        return inference_state


    def _get_mean_std(self, 
                      napari_data, 
                      num_frames,
                      max_num_frames=50
                      ):
        '''
        Given a large video data set, 
        calculate the mean and std of a subset 
        of all frames 
        ''' 
        which_frames = np.linspace(0, num_frames-1, np.min([num_frames, max_num_frames])).astype(int)
        accumulated_frames = np.stack([napari_data[i] for i in which_frames])
        mean = np.mean(accumulated_frames,axis=None)
        std =  np.std(accumulated_frames,axis=None)
        return mean, std