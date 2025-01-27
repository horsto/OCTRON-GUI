from collections import OrderedDict
import numpy as np
import torch
from tqdm.auto import tqdm
import napari
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
        napari_viewer,
        video_layer_idx,
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
        
        self.video_data = napari_data    
        return inference_state


    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx,
        max_frame_num_to_track,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""
        
        # Load current batch of frames into inference state 
        inference_state['images'] = None
        collected_imgs = []
        for img_idx in range(start_frame_idx, start_frame_idx + max_frame_num_to_track + 1):   
            curr_img = torch.from_numpy(self.video_data[img_idx]).permute(2, 0, 1)[torch.newaxis]
            curr_img = self._resize_img(curr_img)
            curr_img = curr_img.float() / 255.0
            # normalize by mean and std
            curr_img -= self.img_mean
            curr_img /= self.img_std
            collected_imgs.append(curr_img)
        collected_imgs = torch.cat(collected_imgs, dim=0)
        inference_state['images'] = collected_imgs  # Add collectred batch back to inference state
        
        
        self.propagate_in_video_preflight(inference_state)

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t
                for obj_output_dict in inference_state["output_dict_per_obj"].values()
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

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # We skip those frames already in consolidated outputs (these are frames
                # that received input clicks or mask). Note that we cannot directly run
                # batched forward on them via `_run_single_frame_inference` because the
                # number of clicks on each object might be different.
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                    
                    # TODO: Reimplement this function 
                    # if self.clear_non_cond_mem_around_input:
                        # # clear non-conditioning memory of the surrounding frames
                        # self._clear_obj_non_cond_mem_around_input(
                        #     inference_state, frame_idx, obj_idx
                        # )
                else:
                    storage_key = "non_cond_frame_outputs"
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
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
                        for objid in inference_state['output_dict_per_obj'].keys():
                            if old_idx in inference_state['output_dict_per_obj'][objid][storage_key]:
                                inference_state['output_dict_per_obj'][objid][storage_key].pop(old_idx)
                            
                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
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
                inference_state, all_pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks