# This file contains helper functions to add layers to the napari viewer through OCTRON
from pathlib import Path
import numpy as np
from napari.utils.notifications import show_info, show_error

def add_sam2_mask_layer(viewer,
                        video_layer,
                        name,
                        project_path,
                        color,
                        ):
    '''
    Generic mask layer for napari and SAM2.
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
    project_path : str or Path
        Path to the project directory.
    base_color : str or list
        Color of the mask layer.
    '''
    project_path = Path(project_path)
    
    assert project_path.exists(), f"Project path {project_path.as_posix()} does not exist."  

    # Check if required metadata exists before creating the dummy mask
    required_keys = ['num_frames', 'height', 'width']
    if all(k in video_layer.metadata for k in required_keys):        
        # Create a numpy memmap file for the mask layer array
        num_frames = video_layer.metadata['num_frames']
        height = video_layer.metadata['height']
        width = video_layer.metadata['width']
        data_shape = (num_frames, height, width)
        memmap_file_path = project_path / f"{name}.dat"
        if memmap_file_path.exists():
            mask_layer_data = np.memmap(memmap_file_path, 
                                        mode='r+', 
                                        dtype=np.uint8, 
                                        shape=data_shape
                                        )
            show_info(f"Mask layer data found at {memmap_file_path.as_posix()}")
        else:
            mask_layer_data = np.memmap(memmap_file_path, 
                                        mode='w+', 
                                        dtype=np.uint8, 
                                        shape=data_shape
                                        )
            mask_layer_data[:] = 0 # All zeros
            mask_layer_data.flush()
    else:
        show_error("Video layer metadata incomplete; dummy mask not created.")
        return
    
    labels_layer = viewer.add_labels(
        mask_layer_data,
        name=name,  
        opacity=0.4,  
        blending='additive',  
        colormap=color, 
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


def add_sam2_shapes_layer(
    viewer,
    name,
    color,
    ):
    '''
    Generic shapes layer for napari and SAM2.
    Initiates the shapes layer, a napari shapes layer instance,
    
    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer object.
    video_layer : napari.layers.Image
        Video layer = video layer object
    name : str
        Name of the new shapes layer.
    base_color : str or list
        Color of the shapes layer.
    '''
    shapes_layer = viewer.add_shapes(None, 
                                 ndim=3,
                                 name=name, 
                                 scale=(1,1),
                                 edge_width=1,
                                 edge_color=color,
                                 face_color=[1,1,1,0],
                                 opacity=.4,
                                 )

    # Hide buttons that you don't want the user to access       
    # TODO: This will be deprecated in future versions of napari.
    qctrl = viewer.window.qt_viewer.controls.widgets[shapes_layer]
    buttons_to_hide = [
                    'line_button',
                    'path_button',
                    'polyline_button',
                    ]
    for btn in buttons_to_hide:
        attr = getattr(qctrl, btn)
        attr.hide()
        
    # Select the current, add tool for the points layer
    viewer.layers.selection.active = shapes_layer
    viewer.layers.selection.active.mode = 'pan_zoom'
    return shapes_layer


def add_sam2_points_layer(    
    viewer,
    name,
    ):
    '''
    Generic points layer for napari and SAM2.
    Initiates the points layer, a napari points layer instance,
    
    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer object.
    video_layer : napari.layers.Image
        Video layer = video layer object
    name : str
        Name of the new points layer.
    '''
    points_layer = viewer.add_points(None, 
                                 ndim=3,
                                 name=name, 
                                 scale=(1,1),
                                 size=40,
                                 border_color=[.7, .7, .7, 1],
                                 border_width=.2,
                                 opacity=.6,
                                 )

    # Select the current, add tool for the points layer
    viewer.layers.selection.active = points_layer
    viewer.layers.selection.active.mode = 'add'
    return points_layer


def add_annotation_projection(    
    viewer,
    name,
    ):
    pass