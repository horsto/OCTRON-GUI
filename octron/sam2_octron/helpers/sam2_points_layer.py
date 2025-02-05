#### Points - > Prediction layer for napari viewer and SAM2 

def add_sam2_points_layer(viewer, 
                          run_new_pred,
                          ):
    '''
    
    
    
    
    '''
    # Add the points layer to the viewer
    points_layer = viewer.add_points(None, 
                                    ndim=3,
                                    name='Annotations', 
                                    scale=(1,1),
                                    size=40,
                                    border_color='dimgrey',
                                    border_width=.2,
                                    opacity=.6,
                                    )
    # Store the initial length of the points data
    previous_length_points = len(points_layer.data)


    left_right_click = 'left'
    
    
    
def on_mouse_press(layer, event):
    '''
    Generic function to catch left and right mouse clicks
    '''
    global left_right_click
    if event.type == 'mouse_press':
        if event.button == 1:  # Left-click
            left_right_click = 'left'
        elif event.button == 2:  # Right-click
            left_right_click = 'right'     
    

def on_points_added(event, 
                    points_layer,
                    labels_layer,
                    previous_length_points,
                    
                    ):
    '''
    Function to run when points are added to the points layer
    '''
    
    global left_right_click
    global prefetcher_worker
    
    current_length = len(points_layer.data)
    if current_length > previous_length_points:
        previous_length_points = current_length 

        # Execute prediction 
        newest_point_data =  points_layer.data[-1]
        if left_right_click == 'left':
            label = 1
            points_layer.face_color[-1] = [0.59607846, 0.98431373, 0.59607846, 1.]
            points_layer.symbol[-1] = 'o'
        elif left_right_click == 'right':
            label = 0
            points_layer.face_color[-1] = [1., 1., 1., 1.]
            points_layer.symbol[-1] = 'x'
        points_layer.refresh() 
        # Run prediction
        frame_idx  = int(newest_point_data[0])
        point_data = newest_point_data[1:][::-1]
        mask = run_new_pred(frame_idx=frame_idx,
                            obj_id=0,
                            label=label,
                            point=point_data,
                            )
        labels_layer.data[frame_idx,:,:] = mask
        labels_layer.refresh()   
        
        # Prefetch batch of images
        # This is done here since adding it as direct mouse interaction 
        # slows down the first prediction
        if not prefetcher_worker.is_running:
            prefetcher_worker.run()


# points_layer.mouse_drag_callbacks.append(on_mouse_press)
# points_layer.events.data.connect(on_points_added)

# # Hide the transform, delete, and select buttons
# qctrl = viewer.window.qt_viewer.controls.widgets[points_layer]
# buttons_to_hide = ['transform_button', 
#                 'delete_button', 
#                 'select_button', 
# ]
# for btn in buttons_to_hide:
#     getattr(qctrl, btn).setVisible(False)
                

# # Select the current, add tool for the points layer
# viewer.layers.selection.active = points_layer
# viewer.layers.selection.active.mode = 'add'
