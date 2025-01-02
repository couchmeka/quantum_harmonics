from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

from storage.data_manager import QuantumDataManager

# Define the notes data
notes_data = [
    # Western 12-Tone System (Octave 3)
    {"note": "C3", "standard": 130.81, "pyth": 128.00, "system": "Western 12-Tone"},
    {"note": "C#3/Db3", "standard": 138.59, "pyth": 136.53, "system": "Western 12-Tone"},
    {"note": "D3", "standard": 146.83, "pyth": 144.00, "system": "Western 12-Tone"},
    {"note": "D#3/Eb3", "standard": 155.56, "pyth": 152.38, "system": "Western 12-Tone"},
    {"note": "E3", "standard": 164.81, "pyth": 162.00, "system": "Western 12-Tone"},
    {"note": "F3", "standard": 174.61, "pyth": 170.67, "system": "Western 12-Tone"},
    {"note": "F#3/Gb3", "standard": 185.00, "pyth": 181.33, "system": "Western 12-Tone"},
    {"note": "G3", "standard": 196.00, "pyth": 192.00, "system": "Western 12-Tone"},
    {"note": "G#3/Ab3", "standard": 207.65, "pyth": 204.80, "system": "Western 12-Tone"},
    {"note": "A3", "standard": 220.00, "pyth": 216.00, "system": "Western 12-Tone"},
    {"note": "A#3/Bb3", "standard": 233.08, "pyth": 228.57, "system": "Western 12-Tone"},
    {"note": "B3", "standard": 246.94, "pyth": 243.00, "system": "Western 12-Tone"},

    # Western 12-Tone System (Octave 4)
    {"note": "C4", "standard": 261.63, "pyth": 256.00, "system": "Western 12-Tone"},
    {"note": "C#4/Db4", "standard": 277.18, "pyth": 273.07, "system": "Western 12-Tone"},
    {"note": "D4", "standard": 293.66, "pyth": 288.00, "system": "Western 12-Tone"},
    {"note": "D#4/Eb4", "standard": 311.13, "pyth": 304.76, "system": "Western 12-Tone"},
    {"note": "E4", "standard": 329.63, "pyth": 324.00, "system": "Western 12-Tone"},
    {"note": "F4", "standard": 349.23, "pyth": 341.33, "system": "Western 12-Tone"},
    {"note": "F#4/Gb4", "standard": 369.99, "pyth": 362.67, "system": "Western 12-Tone"},
    {"note": "G4", "standard": 392.00, "pyth": 384.00, "system": "Western 12-Tone"},
    {"note": "G#4/Ab4", "standard": 415.30, "pyth": 409.60, "system": "Western 12-Tone"},
    {"note": "A4", "standard": 440.00, "pyth": 432.00, "system": "Western 12-Tone"},
    {"note": "A#4/Bb4", "standard": 466.16, "pyth": 457.14, "system": "Western 12-Tone"},
    {"note": "B4", "standard": 493.88, "pyth": 486.00, "system": "Western 12-Tone"},

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
    {"note": "2:1", "base": 432, "ratio": 2.000, "system": "Pythagorean"}
]

system_colors = {
    "Western 12-Tone": "#f9c74f",  # Gold
    "Indian Classical": "#90be6d",  # Green
    "Arabic": "#f94144",  # Red
    "Gamelan": "#577590",  # Blue
    "Pythagorean": "#CBC3E3"  # Purple

}


class MusicTable(QWidget):
    def __init__(self):
        super().__init__()
        self.data_manager = QuantumDataManager()
        self.canvas = None
        self.figure = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create figure and canvas
        self.figure = Figure(figsize=(20, 15))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create the chart immediately
        self.create_chart()

    def create_chart(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Define grid dimensions
        columns = 6
        rows = 9
        # int(np.ceil(len(notes_data) / columns))

        # Loop through notes and plot them
        for i, note in enumerate(notes_data):
            row = i // columns
            col = i % columns
            x = col * 2
            y = -row * 2

            # Draw rectangles for each note
            color = system_colors.get(note["system"], "#cccccc")
            rect = mpatches.Rectangle((x, y), 2, 2, facecolor=color, edgecolor="black")
            ax.add_patch(rect)

            # Add text inside the rectangles
            if note["system"] == "Pythagorean":
                text = f"{note['note']}\n{note['base'] * note['ratio']:.2f} Hz\nRatio: {note['ratio']:.3f}"
            else:
                text = f"{note['note']}\n"
                if note["standard"]:
                    text += f"Std: {note['standard']:.2f}\n"
                if note["pyth"]:
                    text += f"Pyth: {note['pyth']:.2f}"

                # Apply specific styling for the first note in each list (the first item in every system)
            if i % len(notes_data) == 0:  # Check if it's the first item in the list
                ax.text(x + 0.9, y - 0.9, text, ha="center", va="center", fontsize=10,  # Slightly larger font size
                        color="white", fontweight='bold')
            else:
                ax.text(x + 0.9, y - 0.9, text, ha="center", va="center", fontsize=8,
                        color="black" if note["system"] != "Pythagorean" else "white")

        # Adjust the axes
        ax.set_xlim(-1, columns * 2)
        ax.set_ylim(-rows * 2, 1)
        ax.axis("off")

        # Add legend at the bottom with all musical systems in one row
        legend_patches = [mpatches.Patch(color=color, label=system)
                          for system, color in system_colors.items()]
        self.figure.legend(handles=legend_patches,
                           loc="lower center",
                           bbox_to_anchor=(0.5, 0.02),
                           ncol=len(system_colors))

        self.canvas.draw()
