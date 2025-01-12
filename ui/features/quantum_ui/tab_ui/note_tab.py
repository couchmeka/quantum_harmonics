from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget,
    QComboBox,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QGridLayout,
    QSizePolicy,
    QApplication,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from data.backend_data_management.data_manager import QuantumDataManager

notes_data = [
    # Western 12-Tone System (Octave 3)
    {"note": "C1", "standard": 32.70, "pyth": None, "system": "Western 12-Tone"},
    {"note": "C#1/Db1", "standard": 34.65, "pyth": None, "system": "Western 12-Tone"},
    {"note": "D1", "standard": 36.71, "pyth": None, "system": "Western 12-Tone"},
    {"note": "D#1/Eb1", "standard": 38.89, "pyth": None, "system": "Western 12-Tone"},
    {"note": "E1", "standard": 41.20, "pyth": None, "system": "Western 12-Tone"},
    {"note": "F1", "standard": 43.65, "pyth": None, "system": "Western 12-Tone"},
    {"note": "F#1/Gb1", "standard": 46.25, "pyth": None, "system": "Western 12-Tone"},
    {"note": "G1", "standard": 49.00, "pyth": None, "system": "Western 12-Tone"},
    {"note": "G#1/Ab1", "standard": 51.91, "pyth": None, "system": "Western 12-Tone"},
    {"note": "A1", "standard": 55.00, "pyth": None, "system": "Western 12-Tone"},
    {"note": "A#1/Bb1", "standard": 58.27, "pyth": None, "system": "Western 12-Tone"},
    {"note": "B1", "standard": 61.74, "pyth": None, "system": "Western 12-Tone"},
    # Western 12-Tone System (Octave 2)
    {"note": "C2", "standard": 65.41, "pyth": None, "system": "Western 12-Tone"},
    {"note": "C#2/Db2", "standard": 69.30, "pyth": None, "system": "Western 12-Tone"},
    {"note": "D2", "standard": 73.42, "pyth": None, "system": "Western 12-Tone"},
    {"note": "D#2/Eb2", "standard": 77.78, "pyth": None, "system": "Western 12-Tone"},
    {"note": "E2", "standard": 82.41, "pyth": None, "system": "Western 12-Tone"},
    {"note": "F2", "standard": 87.31, "pyth": None, "system": "Western 12-Tone"},
    {"note": "F#2/Gb2", "standard": 92.50, "pyth": None, "system": "Western 12-Tone"},
    {"note": "G2", "standard": 98.00, "pyth": None, "system": "Western 12-Tone"},
    {"note": "G#2/Ab2", "standard": 103.83, "pyth": None, "system": "Western 12-Tone"},
    {"note": "A2", "standard": 110.00, "pyth": None, "system": "Western 12-Tone"},
    {"note": "A#2/Bb2", "standard": 116.54, "pyth": None, "system": "Western 12-Tone"},
    {"note": "B2", "standard": 123.47, "pyth": None, "system": "Western 12-Tone"},
    {"note": "C3", "standard": 130.81, "pyth": 128.00, "system": "Western 12-Tone"},
    {
        "note": "C#3/Db3",
        "standard": 138.59,
        "pyth": 136.53,
        "system": "Western 12-Tone",
    },
    {"note": "D3", "standard": 146.83, "pyth": 144.00, "system": "Western 12-Tone"},
    {
        "note": "D#3/Eb3",
        "standard": 155.56,
        "pyth": 152.38,
        "system": "Western 12-Tone",
    },
    {"note": "E3", "standard": 164.81, "pyth": 162.00, "system": "Western 12-Tone"},
    {"note": "F3", "standard": 174.61, "pyth": 170.67, "system": "Western 12-Tone"},
    {
        "note": "F#3/Gb3",
        "standard": 185.00,
        "pyth": 181.33,
        "system": "Western 12-Tone",
    },
    {"note": "G3", "standard": 196.00, "pyth": 192.00, "system": "Western 12-Tone"},
    {
        "note": "G#3/Ab3",
        "standard": 207.65,
        "pyth": 204.80,
        "system": "Western 12-Tone",
    },
    {"note": "A3", "standard": 220.00, "pyth": 216.00, "system": "Western 12-Tone"},
    {
        "note": "A#3/Bb3",
        "standard": 233.08,
        "pyth": 228.57,
        "system": "Western 12-Tone",
    },
    {"note": "B3", "standard": 246.94, "pyth": 243.00, "system": "Western 12-Tone"},
    # Western 12-Tone System (Octave 4)
    {"note": "C4", "standard": 261.63, "pyth": 256.00, "system": "Western 12-Tone"},
    {
        "note": "C#4/Db4",
        "standard": 277.18,
        "pyth": 273.07,
        "system": "Western 12-Tone",
    },
    {"note": "D4", "standard": 293.66, "pyth": 288.00, "system": "Western 12-Tone"},
    {
        "note": "D#4/Eb4",
        "standard": 311.13,
        "pyth": 304.76,
        "system": "Western 12-Tone",
    },
    {"note": "E4", "standard": 329.63, "pyth": 324.00, "system": "Western 12-Tone"},
    {"note": "F4", "standard": 349.23, "pyth": 341.33, "system": "Western 12-Tone"},
    {
        "note": "F#4/Gb4",
        "standard": 369.99,
        "pyth": 362.67,
        "system": "Western 12-Tone",
    },
    {"note": "G4", "standard": 392.00, "pyth": 384.00, "system": "Western 12-Tone"},
    {
        "note": "G#4/Ab4",
        "standard": 415.30,
        "pyth": 409.60,
        "system": "Western 12-Tone",
    },
    {"note": "A4", "standard": 440.00, "pyth": 432.00, "system": "Western 12-Tone"},
    {
        "note": "A#4/Bb4",
        "standard": 466.16,
        "pyth": 457.14,
        "system": "Western 12-Tone",
    },
    {"note": "B4", "standard": 493.88, "pyth": 486.00, "system": "Western 12-Tone"},
    # Western 12-Tone System (Octave 5)
    {"note": "C5", "standard": 523.25, "pyth": None, "system": "Western 12-Tone"},
    {"note": "C#5/Db5", "standard": 554.37, "pyth": None, "system": "Western 12-Tone"},
    {"note": "D5", "standard": 587.33, "pyth": None, "system": "Western 12-Tone"},
    {"note": "D#5/Eb5", "standard": 622.25, "pyth": None, "system": "Western 12-Tone"},
    {"note": "E5", "standard": 659.26, "pyth": None, "system": "Western 12-Tone"},
    {"note": "F5", "standard": 698.46, "pyth": None, "system": "Western 12-Tone"},
    {"note": "F#5/Gb5", "standard": 739.99, "pyth": None, "system": "Western 12-Tone"},
    {"note": "G5", "standard": 783.99, "pyth": None, "system": "Western 12-Tone"},
    {"note": "G#5/Ab5", "standard": 830.61, "pyth": None, "system": "Western 12-Tone"},
    {"note": "A5", "standard": 880.00, "pyth": None, "system": "Western 12-Tone"},
    {"note": "A#5/Bb5", "standard": 932.33, "pyth": None, "system": "Western 12-Tone"},
    {"note": "B5", "standard": 987.77, "pyth": None, "system": "Western 12-Tone"},
    # Western 12-Tone System (Octave 6)
    {"note": "C6", "standard": 1046.50, "pyth": None, "system": "Western 12-Tone"},
    {"note": "C#6/Db6", "standard": 1108.73, "pyth": None, "system": "Western 12-Tone"},
    {"note": "D6", "standard": 1174.66, "pyth": None, "system": "Western 12-Tone"},
    {"note": "D#6/Eb6", "standard": 1244.51, "pyth": None, "system": "Western 12-Tone"},
    {"note": "E6", "standard": 1318.51, "pyth": None, "system": "Western 12-Tone"},
    {"note": "F6", "standard": 1396.91, "pyth": None, "system": "Western 12-Tone"},
    {"note": "F#6/Gb6", "standard": 1479.98, "pyth": None, "system": "Western 12-Tone"},
    {"note": "G6", "standard": 1567.98, "pyth": None, "system": "Western 12-Tone"},
    {"note": "G#6/Ab6", "standard": 1661.22, "pyth": None, "system": "Western 12-Tone"},
    {"note": "A6", "standard": 1760.00, "pyth": None, "system": "Western 12-Tone"},
    {"note": "A#6/Bb6", "standard": 1864.66, "pyth": None, "system": "Western 12-Tone"},
    {"note": "B6", "standard": 1975.53, "pyth": None, "system": "Western 12-Tone"},
    # Western 12-Tone System (Octave 7)
    {"note": "C7", "standard": 2093.00, "pyth": None, "system": "Western 12-Tone"},
    {"note": "C#7/Db7", "standard": 2217.46, "pyth": None, "system": "Western 12-Tone"},
    {"note": "D7", "standard": 2349.32, "pyth": None, "system": "Western 12-Tone"},
    {"note": "D#7/Eb7", "standard": 2489.02, "pyth": None, "system": "Western 12-Tone"},
    {"note": "E7", "standard": 2637.02, "pyth": None, "system": "Western 12-Tone"},
    {"note": "F7", "standard": 2793.83, "pyth": None, "system": "Western 12-Tone"},
    {"note": "F#7/Gb7", "standard": 2959.96, "pyth": None, "system": "Western 12-Tone"},
    {"note": "G7", "standard": 3135.96, "pyth": None, "system": "Western 12-Tone"},
    {"note": "G#7/Ab7", "standard": 3322.44, "pyth": None, "system": "Western 12-Tone"},
    {"note": "A7", "standard": 3520.00, "pyth": None, "system": "Western 12-Tone"},
    {"note": "A#7/Bb7", "standard": 3729.31, "pyth": None, "system": "Western 12-Tone"},
    {"note": "B7", "standard": 3951.07, "pyth": None, "system": "Western 12-Tone"},
    # Western 12-Tone System (Octave 8)
    {"note": "C8", "standard": 4186.01, "pyth": None, "system": "Western 12-Tone"},
    {"note": "C#8/Db8", "standard": 4434.92, "pyth": None, "system": "Western 12-Tone"},
    {"note": "D8", "standard": 4698.63, "pyth": None, "system": "Western 12-Tone"},
    {"note": "D#8/Eb8", "standard": 4978.03, "pyth": None, "system": "Western 12-Tone"},
    {"note": "E8", "standard": 5274.04, "pyth": None, "system": "Western 12-Tone"},
    {"note": "F8", "standard": 5587.65, "pyth": None, "system": "Western 12-Tone"},
    {"note": "F#8/Gb8", "standard": 5919.91, "pyth": None, "system": "Western 12-Tone"},
    {"note": "G8", "standard": 6271.93, "pyth": None, "system": "Western 12-Tone"},
    {"note": "G#8/Ab8", "standard": 6644.88, "pyth": None, "system": "Western 12-Tone"},
    {"note": "A8", "standard": 7040.00, "pyth": None, "system": "Western 12-Tone"},
    {"note": "A#8/Bb8", "standard": 7458.62, "pyth": None, "system": "Western 12-Tone"},
    {"note": "B8", "standard": 7902.13, "pyth": None, "system": "Western 12-Tone"},
    # Indian Classical (22 Shrutis)
    {"note": "Sa", "standard": 440.00, "pyth": 432.00, "system": "Indian Classical"},
    {"note": "Re♭", "standard": 445.50, "pyth": None, "system": "Indian Classical"},
    {"note": "Re", "standard": 466.16, "pyth": 457.14, "system": "Indian Classical"},
    {"note": "Ga♭", "standard": 471.86, "pyth": None, "system": "Indian Classical"},
    {"note": "Ga", "standard": 493.88, "pyth": 486.00, "system": "Indian Classical"},
    {"note": "Ma", "standard": 523.25, "pyth": 512.00, "system": "Indian Classical"},
    {"note": "Ma#", "standard": 556.88, "pyth": None, "system": "Indian Classical"},
    {"note": "Pa", "standard": 587.33, "pyth": 576.00, "system": "Indian Classical"},
    {"note": "Dha♭", "standard": 593.21, "pyth": None, "system": "Indian Classical"},
    {"note": "Dha", "standard": 622.25, "pyth": 609.52, "system": "Indian Classical"},
    {"note": "Ni♭", "standard": 628.34, "pyth": None, "system": "Indian Classical"},
    {"note": "Ni", "standard": 659.26, "pyth": 648.00, "system": "Indian Classical"},
    # Arabic Quarter Tones
    {"note": "Rast", "standard": 269.40, "pyth": None, "system": "Arabic"},
    {"note": "Duka", "standard": 302.47, "pyth": None, "system": "Arabic"},
    {"note": "Sika", "standard": 339.43, "pyth": None, "system": "Arabic"},
    {"note": "Jaharka", "standard": 359.61, "pyth": None, "system": "Arabic"},
    {"note": "Nawa", "standard": 403.65, "pyth": None, "system": "Arabic"},
    {"note": "Hussaini", "standard": 453.08, "pyth": None, "system": "Arabic"},
    {"note": "Qurdan", "standard": 508.52, "pyth": None, "system": "Arabic"},
    # Gamelan Scales
    {"note": "Slendro 1", "standard": None, "pyth": None, "system": "Gamelan"},
    {"note": "Slendro 2", "standard": None, "pyth": None, "system": "Gamelan"},
    {"note": "Slendro 3", "standard": None, "pyth": None, "system": "Gamelan"},
    {"note": "Slendro 4", "standard": None, "pyth": None, "system": "Gamelan"},
    {"note": "Slendro 5", "standard": None, "pyth": None, "system": "Gamelan"},
    {"note": "Pelog 1", "standard": None, "pyth": None, "system": "Gamelan"},
    {"note": "Pelog 2", "standard": None, "pyth": None, "system": "Gamelan"},
    # Pythagorean Ratios
    {"note": "1:1", "base": 432, "ratio": 1.000, "system": "Pythagorean"},
    {"note": "4:3", "base": 432, "ratio": 1.333, "system": "Pythagorean"},
    {"note": "3:2", "base": 432, "ratio": 1.500, "system": "Pythagorean"},
    {"note": "φ:1", "base": 432, "ratio": 1.618, "system": "Pythagorean"},
    {"note": "2:1", "base": 432, "ratio": 2.000, "system": "Pythagorean"},
]

system_colors = {
    "Western 12-Tone": "#f9c74f",  # Gold
    "Indian Classical": "#90be6d",  # Green
    "Arabic": "#f94144",  # Red
    "Gamelan": "#577590",  # Blue
    "Pythagorean": "#CBC3E3",  # Purple
}


class MusicTable(QWidget):
    def __init__(self):
        super().__init__()
        self.system_selector = None
        self.table = None
        self.init_ui()

    def init_ui(self):
        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Dropdown for system selection
        self.system_selector = QComboBox()
        self.system_selector.addItems(
            ["Western 12-Tone", "Indian Classical", "Arabic", "Gamelan", "Pythagorean"]
        )
        self.system_selector.currentTextChanged.connect(self.update_table)
        self.main_layout.addWidget(self.system_selector, alignment=Qt.AlignHCenter)

        # Table for displaying notes
        self.table = QTableWidget()
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.itemClicked.connect(
            self.copy_note_to_clipboard
        )  # Connect item click signal
        self.main_layout.addWidget(self.table)

        # Style the table
        self.table.setStyleSheet(
            """
            QTableWidget {
                background-color: #f9f9f9;
                gridline-color: #333;
                font-size: 14px;
            }
            QTableWidget::item {
                padding: 10px;
            }
            QHeaderView::section {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }
            """
        )

        # Initialize table with the first system
        self.update_table(self.system_selector.currentText())

        # Set the main layout
        self.setLayout(self.main_layout)

    def update_table(self, selected_system):
        # Filter notes based on selected system
        filtered_notes = [
            note for note in notes_data if note["system"] == selected_system
        ]

        # Determine table dimensions
        columns = 10  # Fixed number of columns
        rows = (len(filtered_notes) + columns - 1) // columns

        # Configure table
        self.table.setRowCount(rows)
        self.table.setColumnCount(columns)
        self.table.clear()

        # Populate table with notes
        for i, note in enumerate(filtered_notes):
            row = i // columns
            col = i % columns

            text = f"{note['note']}\n"
            if note.get("standard"):
                text += f"Std: {note['standard']:.2f}\n"
            if note.get("pyth"):
                text += f"Pyth: {note['pyth']:.2f}"

            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)

            # Apply alternating row colors
            if row % 2 == 0:
                item.setBackground(QColor("#E8F5E9"))  # Light green
            else:
                item.setBackground(QColor("#FFEBEE"))  # Light red

            # Set the item in the table
            self.table.setItem(row, col, item)

        # Resize table to fit content
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()

    def copy_note_to_clipboard(self, item):
        """Extracts the note from the clicked item and copies it to the clipboard."""
        full_text = item.text()
        # Extract just the note (first line)
        note = full_text.split("\n")[0]

        # Copy to clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(note)

        print(f"Copied to clipboard: {note}")  # Optional: For debugging
