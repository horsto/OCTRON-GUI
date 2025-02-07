# Create a continous colormap and cmap_range for each label 

import numpy as np
import cmasher as cmr

def create_label_colors(cmap='cmr.tropical', label_n=4, obj_n=5):
    '''
    Create color map dictionary for labels 
    label(int) -> color list -> color(4D)
    
    For each label (n=label_cat_n) create a sub colormap with color_cat_n colors.
    
    '''

    label_cat_rel = np.linspace(0,1,label_n+1) # This must be the ugliest written fctn in the world

    label_colors = {}
    for cat in range(len(label_cat_rel)-1):
        rel_range = label_cat_rel[cat:cat+2]
        colors = np.array(cmr.take_cmap_colors(cmap, 
                                              obj_n, 
                                              cmap_range=(rel_range[0], rel_range[1]), 
                                              return_fmt='int'
                                              ) 
                        ) / 255.0
        colors4d = np.hstack([colors, np.ones((len(colors), 1))])
        label_colors[cat] = {i+1: color for i, color in enumerate(colors4d)} # Keys start at 1 !
        label_colors[cat][None] = np.array([0,0,0,0]).astype(np.float32)
    return label_colors