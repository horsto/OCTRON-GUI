### Generic mask layer for napari and SAM2
import numpy as np
from napari.utils import DirectLabelColormap

def add_sam2_mask_layer(num_frames,
                        video_height,
                        video_width,
                        viewer,
                        label_id,
                        obj_id,
                        colors,
                        ):
    # Instantiate the mask and annotation layers 
    # Keep them empty at start 
    mask_layer_dummy = np.zeros((num_frames, video_height, video_width), dtype=np.uint8)
    mask_layer_dummy.shape


    # Select colormap for labels layer based on category (label) and current object ID 
    current_color_labelmap = DirectLabelColormap(color_dict=colors[label_id], 
                                                use_selection=True, 
                                                selection=obj_id+1,
                                                )
    labels_layer = viewer.add_labels(
        mask_layer_dummy, 
        name='Mask',  # Name of the layer
        opacity=0.4,  # Optional: opacity of the labels
        blending='additive',  # Optional: blending mode
        colormap=current_color_labelmap, 
    )

    qctrl = viewer.window.qt_viewer.controls.widgets[labels_layer]
    buttons_to_hide = ['erase_button',
                    'fill_button',
                    'paint_button',
                    'pick_button',
                    'polygon_button',
                    'transform_button',
                    ]
    for btn in buttons_to_hide:
        getattr(qctrl, btn).setEnabled(False)