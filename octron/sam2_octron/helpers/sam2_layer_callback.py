# OCTRON SAM2 related callbacks
import numpy as np

from octron.sam2_octron.helpers.sam2_octron import run_new_pred


def on_shapes_changed(event,
                      viewer,
                      video_height,
                      video_width,
                      predictor,
                      shapes_layer,
                      labels_layer,
                      prefetcher_worker,
                      ):
    '''
    Callback function for napari shapes layer.
    This function is called whenever changes are made to the shapes layer.
    It extracts the mask from the shapes layer and runs the predictor on it.
    There is one special case for the rectangle tool, which acts as "box input" 
    to SAM2 instead of creating an input mask.
    
    '''
    action = event.action
    if action in ['added','removed','changed']:
        frame_idx = viewer.dims.current_step[0] 
        
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
                                obj_id=0,
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
                                obj_id=0,
                                labels=label,
                                masks=shape_mask,
                                )

        labels_layer.data[frame_idx] = mask
        labels_layer.refresh()
        
        # Prefetch batch of images
        if not prefetcher_worker.is_running:
            prefetcher_worker.run()
    return