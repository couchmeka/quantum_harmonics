# styles.py
from PyQt6.QtWidgets import QGroupBox, QPushButton

# Common color schemes
# styles.py

COLORS = {
    'primary': '#3d405e',
    'secondary': 'rgba(25, 118, 210, 0.3)',
    'background': '#1f2f4a',
    'surface': 'rgba(89, 92, 120, 0.6)',
    'text': '#ffffff',
    'accent': '#2196F3',
    'error': '#f94144',
    'success': '#90be6d',
    'warning': '#f9c74f'
}

# Base text style
base_style = f"""
    color: {COLORS['text']};
    font-family: Arial, sans-serif;
"""

# Button style
button_style = f"""
    QPushButton {{
        background-color: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['primary']};
        border-radius: 5px;
        padding: 10px;
        font-size: 12px;
        min-width: 100px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['secondary']};
        border: 1px solid #1976D2;
   }}
    QPushButton:pressed {{
                background-color: rgba(13, 71, 161, 0.3);
            
    }}
"""

# Slider style
slider_style = f"""
    QSlider::groove:horizontal {{
        border: 1px solid {COLORS['primary']};
        height: 8px;
        background: {COLORS['surface']};
        margin: 2px 0;
        border-radius: 4px;
    }}
    QSlider::handle:horizontal {{
        background: {COLORS['text']};
        border: none;
        width: 18px;
        margin: -6px 0;
        border-radius: 9px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {COLORS['accent']};
    }}
"""

# Combo box style
combobox_style = f"""
    QComboBox {{
        background-color: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['primary']};
        border-radius: 4px;
        padding: 5px;
        min-width: 150px;
        font-family: Arial, sans-serif;
    }}
    QComboBox:hover {{
        border-color: {COLORS['secondary']};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    }}
    QComboBox QAbstractItemView {{
        background-color: {COLORS['background']};
        color: {COLORS['text']};
        selection-background-color: {COLORS['secondary']};
        border: 1px solid {COLORS['primary']};
    }}
"""


# Group box style
groupbox_style = f"""
    QGroupBox {{
        color: {COLORS['text']};
        font-size: 14px;
        font-weight: bold;
        border: 1px solid {COLORS['primary']};
        border-radius: 5px;
        padding: 15px;
        background-color: {COLORS['background']};
        font-family: Arial, sans-serif;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        top: 5px;
    }}
"""

# Text edit style
textedit_style = f"""
    QTextEdit {{
        background-color: {COLORS['surface']};
        color: {COLORS['text']};
        border: 1px solid {COLORS['primary']};
        border-radius: 4px;
        padding: 5px;
        font-family: Arial, sans-serif;
        font-size: 12px;
    }}
    QTextEdit:focus {{
        border-color: {COLORS['secondary']};
    }}
"""

# Line edit style
# Fix the input styles to be dark with light text
lineedit_style = f"""
QLineEdit {{
    background-color: white;
    color: black;
    border: 1px solid {COLORS['primary']};
    border-radius: 4px;
    padding: 5px;
    font-family: Arial, sans-serif;
}}
"""


# Layout settings
DEFAULT_MARGINS = (20, 20, 20, 20)
DEFAULT_SPACING = 10


# Helper functions
def create_group_box(title):
    """Create a styled group box"""
    group = QGroupBox(title)
    group.setStyleSheet(groupbox_style)
    return group


def create_button(text, icon=None):
    """Create a styled button"""
    button = QPushButton(text)
    if icon:
        button.setIcon(icon)
    button.setStyleSheet(button_style)
    return button
