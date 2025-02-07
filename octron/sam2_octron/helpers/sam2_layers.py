# This file contains helper functions to add layers to the napari viewer through OCTRON


def add_sam2_mask_layer(viewer,
                        video_layer,
                        name,
                        base_color,
                        ):
    '''
    Generic mask layer for napari and SAM2
    Initiates the mask layer, a napari labels layer instance,
    and fixes it's color to "base_color'.
    
    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer object.
    video_layer : napari.layers.Image
        Video layer = video layer object
    name : str
        Name of the new mask layer.
    base_color : str or list
        Color of the mask layer.
    '''
    labels_layer = viewer.add_labels(
        video_layer.metadata['dummy'], 
        name=name,  
        opacity=0.4,  
        blending='additive',  
        colormap=base_color, 
    )

    # Hide buttons that you don't want the user to access
    # TODO: This will be deprecated in future versions of napari.
    qctrl = viewer.window.qt_viewer.controls.widgets[labels_layer]
    buttons_to_hide =  ['erase_button',
                        'fill_button',
                        'paint_button',
                        'pick_button',
                        'polygon_button',
                        'transform_button',
                        ]
    for btn in buttons_to_hide: 
        getattr(qctrl, btn).hide() 
        
    return labels_layer