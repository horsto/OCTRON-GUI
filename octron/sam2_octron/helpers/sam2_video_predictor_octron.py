from collections import OrderedDict

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
        images,
    ):
        assert len(images.shape) == 4, f"images should have shape (num_frames, 3, H, W), got {images.shape}"
        """Initialize an inference state."""
        compute_device = self.device  # device of the model


        # Generic torch resize transformation
        self._resize_img = Resize(
                        size=(self.image_size)
                    )


        inference_state = {}
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] =  images.shape[2]
        inference_state["video_width"] =   images.shape[3]
        
 
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        
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
