# OCTRON SAM2 related callbacks
import time
import numpy as np
from skimage import measure

from napari.qt.threading import thread_worker

from octron.sam2_octron.helpers.sam2_octron import (SAM2_octron,
                                                    run_new_pred,
)

class sam2_octron_callbacks():
    '''
    Callback for octron and SAM2.
    '''
    def __init__(self, octron):
        # Store the reference to the main OCTRON widget
        self.octron = octron
        self.viewer = octron._viewer
        self.left_right_click = None
            
    def on_mouse_press(self, layer, event):
        '''
        Generic function to catch left and right mouse clicks
        '''
        if event.type == 'mouse_press':
            if event.button == 1:  # Left-click
                self.left_right_click = 'left'
            elif event.button == 2:  # Right-click
                self.left_right_click = 'right'     
        
    
    def on_shapes_changed(self, event):
        '''
        Callback function for napari annotation "Shapes" layer.
        This function is called whenever changes are made to annotation shapes.
        It extracts the mask from the shapes layer and runs the predictor on it.
        
        There is one special case for the "rectangle tool", which acts as "box input" 
        to SAM2 instead of creating an input mask.
        
        '''
        
        
        action = event.action
        if action in ['added','removed','changed']:
            frame_idx = self.viewer.dims.current_step[0] 
            shapes_layer = event.source
            obj_id = shapes_layer.metadata['_obj_id']
            
            # Get the corresponding mask layer 
            organizer_entry = self.octron.object_organizer.entries[obj_id]
            mask_layer = organizer_entry.mask_layer
            if mask_layer is None:
                # That should actually never happen 
                self.octron.show_error('No corresponding mask layer found.')
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
                top_left, _, bottom_right, _ = box
                top_left, bottom_right = top_left[1:], bottom_right[1:]
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
                shape_mask = shapes_layer.to_masks((video_height, video_width))
                shape_mask = np.sum(shape_mask, axis=0)
                if not isinstance(shape_mask, np.ndarray):
                    return
                shape_mask[shape_mask > 0] = 1
                shape_mask = shape_mask.astype(np.uint8)
            
                label = 1 # Always positive for now
                mask = run_new_pred(predictor=predictor,
                                    frame_idx=frame_idx,
                                    obj_id=obj_id,
                                    labels=label,
                                    masks=shape_mask,
                                    )

            mask_layer.data[frame_idx] = mask
            mask_layer.refresh()
            organizer_entry.add_predicted_frame(frame_idx)
            # Prefetch next batch of images
            if not self.octron.prefetcher_worker.is_running:
                self.octron.prefetcher_worker.run()
                
        else:
            # Catching all above with ['added','removed','changed']
            pass
        return
    
    
    def on_points_changed(self, event):
        '''
        Callback function for napari annotation "Points" layer.
        This function is called whenever changes are made to annotation points.

        '''
        action = event.action
        
        left_positive_color  = [0.59607846, 0.98431373, 0.59607846, 1.]
        right_negative_color = [1., 1., 1., 1.]
        
        frame_idx  = self.viewer.dims.current_step[0] 
        points_layer = event.source
        obj_id = points_layer.metadata['_obj_id']
        
        predictor = self.octron.predictor
        
        # Get the corresponding mask layer 
        organizer_entry = self.octron.object_organizer.entries[obj_id]
        mask_layer = organizer_entry.mask_layer
        if mask_layer is None:
            # That should actually never happen 
            self.octron.show_error('No corresponding mask layer found.')
            return    
        
        if action == 'added':
            # A new point has just been added. 
            # Find out if you are dealing with a left or right click    
            if self.left_right_click == 'left':
                label = 1
                points_layer.face_color[-1] = left_positive_color
                points_layer.symbol[-1] = 'o'
            elif self.left_right_click == 'right':
                label = 0
                points_layer.face_color[-1] = right_negative_color
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
                                obj_id=0,
                                labels=labels,
                                points=point_data,
                                )
            mask_layer.data[frame_idx,:,:] = mask
            mask_layer.refresh()  
            organizer_entry.add_predicted_frame(frame_idx)
        else:
            # Catching all above with ['added','removed','changed']
            pass
        return    
    
    
    
    
    def batch_predict(self):
        '''
        Threaded function to run the predictor on a batch of frames.
        Uses SAM2 => propagate_in_video function.
        
        
        '''
        # TODO: Find obj_id 
        # THIS WHOLE THING IS NOT OBJECT SPECIFIC !!!!!
        
        
        # Make sure user cannot click twice 
        self.octron.predict_next_batch_btn.setEnabled(False)
        
        
        
        current_obj_id = 0 
        max_imgs = self.octron.chunk_size
        
        frame_idx = self.viewer.dims.current_step[0]        
        video_segments = {} 
        start_time = time.time()
        # Prefetch images if they are not cached yet 
        _ = self.octron.predictor.images[slice(frame_idx,frame_idx+max_imgs)]
        
        # Loop over frames and run prediction (single frame!)
        frame_counter = 0 
        for out_frame_idx, out_obj_ids, out_mask_logits in self.octron.predictor.propagate_in_video(start_frame_idx=frame_idx, 
                                                                                                    max_frame_num_to_track=max_imgs):
            
            for i, out_obj_id in enumerate(out_obj_ids):
                
                torch_mask = out_mask_logits[i] > 0.0
                out_mask = torch_mask.cpu().numpy()

                video_segments[out_frame_idx] = {out_obj_id: out_mask}
                if not out_obj_id in self.octron.predictor.inference_state['centroids']:
                    self.octron.predictor.inference_state['centroids'][out_obj_id] = {}
                if not out_obj_id in self.octron.predictor.inference_state['areas']:
                    self.octron.predictor.inference_state['areas'][out_obj_id] = {}
            
            self.octron.predicted_frame_indices[current_obj_id][out_frame_idx] = 1  
                  
            # PICK ONE OBJ (OBJ_ID = 0 or whatever)
            #  Add the mask image as a new labels layer
            mask = video_segments[out_frame_idx][current_obj_id] # THIS NEEDS TO BE MADE LAYER SPECIFIC 
            current_label = current_obj_id+1
            if len(np.unique(mask))>1:
                mask[mask==np.unique(mask)[1]] = current_label 
            mask = mask.squeeze()
            
            self.octron.obj_id_layer[current_obj_id].data[out_frame_idx,:,:] = mask
            self.viewer.dims.set_point(0,out_frame_idx)
            self.octron.obj_id_layer[current_obj_id].refresh()
            
            props = measure.regionprops(mask.astype(int))[0]
            self.octron.predictor.inference_state['centroids'][current_obj_id][out_frame_idx] = props.centroid
            self.octron.predictor.inference_state['areas'][current_obj_id][out_frame_idx] = props.area
            frame_counter += 1
            # Update octron progress bar
            self.octron.batch_predict_progressbar.setValue(frame_counter)
        
            
        end_time = time.time()
        print(f'start idx {frame_idx} | {max_imgs} frames in {end_time-start_time} s')
            
        # Re-enable the predict button
        self.octron.predict_next_batch_btn.setEnabled(True)
        # Update octron progress bar
        self.octron.batch_predict_progressbar.setValue(0)