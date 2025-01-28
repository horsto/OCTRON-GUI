'''
OCTRON



'''
import os 
from pathlib import Path
cur_path = Path(os.path.abspath(__file__)).parent

from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

import napari
import napari.window
from napari.utils.notifications import show_info


# SAM2 specific 
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
from octron.sam2_octron.helpers.build_sam2_octron import build_sam2_octron  


from qtpy.QtWidgets import QWidget

class octron_widget(QWidget):
    '''
    
    '''
    def __init__(self, viewer: 'napari.viewer.Viewer', parent=None):
        super().__init__(parent)
        
        
        self._viewer = viewer
        
        # Create a button
        btn = QPushButton("Click me 🐙")
        # Connect the click event to a function
        btn.clicked.connect(self._on_click)
        btn.setFixedWidth(150)


        # Forwards backwards buttons 

        forw_btn = QPushButton("Next")
        forw_btn.clicked.connect(self._forw_1frame)
        forw_btn.setFixedWidth(50)
        


        self.setLayout(QHBoxLayout())
        # add it to the layout
        self.layout().addWidget(btn)
        self.layout().addWidget(forw_btn)
        
        self.predictor, self.device = self._initialize_sam2()
        
        
    def _initialize_sam2(self):
        '''
        Initialize the SAM2 model
        '''
        sam2_folder = Path('sam2_octron')
        
        
        # TODO: Make checkpoint path and config file path configurable  
        checkpoint = 'sam2.1_hiera_large.pt' # under folder /checkpoints
        model_cfg = 'sam2.1/sam2.1_hiera_l.yaml' # under folder /configs
        # ------------------------------------------------------------------------------------
        sam2_checkpoint = cur_path / sam2_folder / Path(f'checkpoints/{checkpoint}')
        model_cfg = Path(f'configs/{model_cfg}')
        
        assert sam2_checkpoint.exists(), f'Checkpoint file does not exist: {sam2_checkpoint}'
        assert (cur_path/sam2_folder/model_cfg).exists(), f'Config file does not exist: {cur_path/sam2_folder/model_cfg}'
        predictor, device  = build_sam2_video_predictor_octron(config_file=model_cfg.as_posix(), 
                                                               ckpt_path=sam2_checkpoint.as_posix(), 
                                                               )
                
        return predictor, device    
    
    def _on_click(self):
        show_info('Hello and welcome to OCTRON!\nOctopuses are amazing creatures 🐙')
        
    def _forw_1frame(self):
        current_indices = self._viewer.dims.current_step
        self._viewer.dims.set_point(0,current_indices[0]+100)