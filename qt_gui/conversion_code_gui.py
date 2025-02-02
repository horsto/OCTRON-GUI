# Code to correct the octron.py file
# that is created when running
# uic -g python octron.ui > octron.py

# Rational: The octron.py contains some errors that need to be corrected.
# Some of the changes that need to be made are:
# - Replace PySide2 imports with qtpy imports
# - Replace "octron_widgetui" with "self" when used as a parameter
# - Add "qt_gui/" in front of every SVG path

import re

# Define the file paths
input_file_path = 'octron.py'
output_file_path = 'octron_corrected.py'

# Read the content of the input file
with open(input_file_path, 'r') as file:
    content = file.read()

# Replace PySide2 imports with qtpy imports
content = re.sub(r'from PySide2.QtCore import \*', 'from qtpy.QtCore import *', content)
content = re.sub(r'from PySide2.QtGui import \*', 'from qtpy.QtGui import *', content)
content = re.sub(r'from PySide2.QtWidgets import \*', 'from qtpy.QtWidgets import *', content)

# Modify the setupUi method signature
content = re.sub(r'def setupUi\(self, octron_widgetui\)', 'def setupUi(self)', content)

# Replace "octron_widgetui" with "self" when used as a parameter
content = re.sub(r'\boctron_widgetui\b', 'self', content)

# Add "qt_gui/" within every SVG path in the strings
content = re.sub(r'(")([^"]*\.svg")', r'\1qt_gui/\2', content)

# Delete any lines containing "connectSlotsByName"
content = re.sub(r'^.*connectSlotsByName.*$\n?', '', content, flags=re.MULTILINE)

# Remove the second 'self' in retranslateUi method
content = re.sub(r'def retranslateUi\(self, self\)', 'def retranslateUi(self)', content)

# Change 'self.retranslateUi(octron_widgetui)' to 'self.retranslateUi()'
content = re.sub(r'self.retranslateUi\(self\)', 'self.retranslateUi()', content)

# Write the corrected content to the output file
with open(output_file_path, 'w') as file:
    file.write(content)

print(f"Corrected code has been written to {output_file_path}")