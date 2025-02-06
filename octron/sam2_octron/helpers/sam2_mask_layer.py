### Generic mask layer for napari and SAM2

def add_sam2_mask_layer(viewer,
                        image_layer,
                        name,
                        base_color,
                        ):
    '''
    Iniiates the mask layer and fixes it's color to "base_color'.
    
    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer object.
    image_layer : napari.layers.Image
        Image layer = video layer object
    name : str
        Name of the mask layer.
    base_color : str or list
        Color of the mask layer.
    '''
    labels_layer = viewer.add_labels(
        image_layer['mask_dummy'], 
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