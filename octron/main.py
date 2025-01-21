'''
OCTRON



'''
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

import napari
import napari.window
from napari.utils.notifications import show_info


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
        

    def _on_click(self):
        show_info('Hello and welcome to OCTRON!\nOctopuses are amazing creatures 🐙')
        
    def _forw_1frame(self):
        current_indices = self._viewer.dims.current_step
        self._viewer.dims.set_point(0,current_indices[0]+100)