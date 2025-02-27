# OCTRON SAM2 related callbacks
import time
import numpy as np
from napari.utils.notifications import (
    show_info, 
    show_error,
)
from octron.sam2_octron.helpers.sam2_octron import (
    SAM2_octron,
    run_new_pred,
)
import warnings 
warnings.simplefilter("ignore")

class sam2_octron_callbacks():
    """
    Callback for octron and SAM2.
    """
    def __init__(self, octron):
        # Store the reference to the main OCTRON widget
        self.octron = octron
        self.viewer = octron._viewer
        self.left_right_click = None
            
    def on_mouse_press(self, layer, event):
        """
        Generic function to catch left and right mouse clicks
        """
        if event.type == 'mouse_press':
            if event.button == 1:  # Left-click
                self.left_right_click = 'left'
            elif event.button == 2:  # Right-click
                self.left_right_click = 'right'     
        
    
    def on_shapes_changed(self, event):
        """
        Callback function for napari annotation "Shapes" layer.
        This function is called whenever changes are made to annotation shapes.
        It extracts the mask from the shapes layer and runs the predictor on it.
        
        There is one special case for the "rectangle tool", which acts as "box input" 
        to SAM2 instead of creating an input mask.
        
        """
        
        
        action = event.action
        if action in ['added','removed','changed']:
            frame_idx = self.viewer.dims.current_step[0] 
            shapes_layer = event.source
            obj_id = shapes_layer.metadata['_obj_id']
            
            # Get the corresponding mask layer 
            organizer_entry = self.octron.object_organizer.entries[obj_id]
            prediction_layer = organizer_entry.prediction_layer
            if prediction_layer is None:
                # That should actually never happen 
                self.octron.show_error('No corresponding prediction layer found.')
                return   
            
            video_height = self.octron.video_layer.metadata['height']    
            video_width = self.octron.video_layer.metadata['width']   
            predictor = self.octron.predictor
            
            ############################################################    
            
            if shapes_layer.mode == 'add_rectangle':
                if action == 'removed':
                    return
                # Take care of box input first. 
                # If the rectangle tool is selected, extract "box" coordinates
                box = shapes_layer.data[-1]
                if len(box) > 4:
                    box = box[-4:]
                # Find out what the top left and bottom right coordinates are
                box = np.stack(box)[:,1:]
                box_sum = np.sum(box, axis=1)
                top_left_idx = np.argmin(box_sum, axis=0)
                bottom_right_idx = np.argmax(box_sum, axis=0)
                top_left, bottom_right = box[top_left_idx,:], box[bottom_right_idx,:]
                
                
                mask = run_new_pred(predictor=predictor,
                                    frame_idx=frame_idx,
                                    obj_id=obj_id,
                                    labels=[1],
                                    box=[top_left[1],
                                         top_left[0],
                                         bottom_right[1],
                                         bottom_right[0]
                                         ],
                                    )
                shapes_layer.data = shapes_layer.data[:-1]
                shapes_layer.refresh()  
                
            else:
                # In all other cases, just treat shapes as masks 
                shape_masks = np.stack(shapes_layer.to_masks((video_height, video_width)))
                if len(shape_masks) == 1: 
                    shape_mask = shape_masks[0]
                else:
                    frame_indices = np.array([s[0][0] for s in shapes_layer.data]).astype(int)
                    valid_indices = np.argwhere(frame_indices == frame_idx)
                    valid_masks = shape_masks[valid_indices].squeeze()
                    if valid_masks.ndim == 3:
                        shape_mask = np.sum(valid_masks, axis=0)
                    else:
                        shape_mask = valid_masks
                shape_mask[shape_mask > 0] = 1
                shape_mask = shape_mask.astype(np.uint8)
            
                label = 1 # Always positive for now
                mask = run_new_pred(predictor=predictor,
                                    frame_idx=frame_idx,
                                    obj_id=obj_id,
                                    labels=label,
                                    masks=shape_mask,
                                    )

            prediction_layer.data[frame_idx] = mask
            prediction_layer.refresh()
            # Prefetch next batch of images
            if not self.octron.prefetcher_worker.is_running:
                self.octron.prefetcher_worker.run()
                
        else:
            # Catching all above with ['added','removed','changed']
            pass
        return
    
    
    def on_points_changed(self, event):
        """
        Callback function for napari annotation "Points" layer.
        This function is called whenever changes are made to annotation points.

        """
        action = event.action
        predictor = self.octron.predictor
        frame_idx  = self.viewer.dims.current_step[0] 
        points_layer = event.source
        obj_id = points_layer.metadata['_obj_id']
        
        # Get the corresponding mask layer 
        organizer_entry = self.octron.object_organizer.entries[obj_id]
        prediction_layer = organizer_entry.prediction_layer
        color = organizer_entry.color
        
        if prediction_layer is None:
            # That should actually never happen 
            self.octron.show_error('No corresponding prediction (mask) layer found.')
            return    
        
        if action == 'added':
            # A new point has just been added. 
            # Find out if you are dealing with a left or right click    
            if self.left_right_click == 'left':
                label = 1
                points_layer.face_color[-1] = color
                points_layer.border_color[-1] = [.7, .7, .7, 1]
                points_layer.symbol[-1] = 'o'
            elif self.left_right_click == 'right':
                label = 0
                points_layer.face_color[-1] = [.7, .7, .7, 1]
                points_layer.border_color[-1] = color 
                points_layer.symbol[-1] = 'x'
            points_layer.refresh() # THIS IS IMPORTANT
            # Prefetch next batch of images
            if not self.octron.prefetcher_worker.is_running:
                self.octron.prefetcher_worker.run()
            
        # Loop through all the data and create points and labels
        if action in ['added','removed','changed']:
            labels = []
            point_data = []
            for pt_no, pt in enumerate(points_layer.data):
                if pt[0] != frame_idx:  
                    continue
                # Find out which label was attached to the point
                # by going through the symbol lists
                cur_symbol = points_layer.symbol[pt_no]
                if cur_symbol in ['o','disc']:
                    label = 1
                else:
                    label = 0
                labels.append(label)
                point_data.append(pt[1:][::-1]) # index 0 is the frame number
                
            # Then run the actual prediction
            mask = run_new_pred(predictor=predictor,
                                frame_idx=frame_idx,
                                obj_id=obj_id,
                                labels=labels,
                                points=point_data,
                                )
            prediction_layer.data[frame_idx,:,:] = mask
            prediction_layer.refresh()  
        else:
            # Catching all above with ['added','removed','changed']
            pass
        return    
    
    
    def prefetch_images(self):
        """
        Thread worker for prefetching images for fast processing in the viewer
        """
        predictor = self.octron.predictor
        assert predictor, "No model loaded."
        assert predictor.is_initialized, "Model not initialized."
        
        viewer = self.octron._viewer    
        video_layer = self.octron.video_layer   
        num_frames = video_layer.metadata['num_frames']
        # Chunk size and skipping of frames
        chunk_size = self.octron.chunk_size
        skip_frames = self.octron.skip_frames   
        
        current_indices = viewer.dims.current_step
        current_frame = current_indices[0]
        
        # Create slice and check if there are enough frames to prefetch
        end_frame = min(num_frames-1, current_frame + chunk_size * skip_frames)
        image_indices = list(range(current_frame, end_frame, skip_frames))
        if not image_indices:
            return
        
        print(f'⚡️ Prefetching {len(image_indices)} images, skipping {skip_frames} frames | start: {current_frame}')
        _ = predictor.images[image_indices]

    
    def next_predict(self):
        """
        Threaded function to run the predictor forward on exactly one frame.
        Uses SAM2 => propagate_in_video function.
        
        """    

        # Prefetch images if they are not cached yet 
        # For this, reset the chunk_size to 1
        # This will ensure we are only prefetching one frame
        # At the end of the function, we will reset the chunk_size to the original value
        skip_frames = self.octron.skip_frames_spinbox.value()
        if skip_frames < 1:
            skip_frames = 1 # Just hard reset any unrealistic values here
        self.octron.skip_frames = skip_frames  
        chunk_size_real = self.octron.chunk_size
        self.octron.chunk_size = 1
        if getattr(self.octron.predictor, 'images', None) is not None:
            self.prefetch_images()
        
        current_frame = self.viewer.dims.current_step[0]         
        num_frames = self.octron.video_layer.metadata['num_frames']
        end_frame = min(num_frames-1, current_frame + self.octron.chunk_size * skip_frames)
        image_idxs = [current_frame, end_frame]
        
        start_time = time.time()    
        # Just copying routine here from the batch_predict function    
        # Loop over frames and run prediction (single frame!)
        counter = 1
        for out_frame_idx, out_obj_ids, out_mask_logits in self.octron.predictor.propagate_in_video(
            processing_order=image_idxs
        ):
            
            if counter == 1:
                last_run = True
            else:
                last_run = False
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0).cpu().numpy().astype(np.uint8)
                yield counter, out_frame_idx, out_obj_id, mask.squeeze(), last_run

            counter += 1
            
        end_time = time.time()
        print(f'Start idx {current_frame} | Predicted 1 frame in {end_time-start_time:.2f} seconds')
        
        self.octron.chunk_size = chunk_size_real
        return
    
    
    
    
    def batch_predict(self):
        """
        Threaded function to run the predictor forward on a batch of frames.
        Uses SAM2 => propagate_in_video function.
        
        """
        

        skip_frames = self.octron.skip_frames_spinbox.value()
        if skip_frames < 1:
            skip_frames = 1 # Just hard reset any unrealistic values here
        self.octron.skip_frames = skip_frames  

        # Prefetch images if they are not cached yet 
        if getattr(self.octron.predictor, 'images', None) is not None:
            self.prefetch_images()
        
        current_frame = self.viewer.dims.current_step[0]         
        num_frames = self.octron.video_layer.metadata['num_frames']
        end_frame = min(num_frames-1, current_frame + self.octron.chunk_size * skip_frames)
        image_idxs = list(range(current_frame, end_frame, skip_frames)) 
        start_time = time.time()        
        # Loop over frames and run prediction (single frame!)
        counter = 1
        for out_frame_idx, out_obj_ids, out_mask_logits in self.octron.predictor.propagate_in_video(
            processing_order=image_idxs
            ):
            
            if counter == end_frame:
                last_run = True
            else:
                last_run = False
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0).cpu().numpy().astype(np.uint8)
                yield counter, out_frame_idx, out_obj_id, mask.squeeze(), last_run

            counter += 1
            
        end_time = time.time()
        print(f'Start idx {current_frame} | Predicted {self.octron.chunk_size} frames in {end_time-start_time:.2f} seconds')
        
        return