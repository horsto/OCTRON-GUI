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
        btn = QPushButton("Click me")
        # Connect the click event to a function
        btn.clicked.connect(self._on_click)


        self.setLayout(QHBoxLayout())
        # add it to the layout
        self.layout().addWidget(btn)

    def _on_click(self):
        show_info('Hello, world!')