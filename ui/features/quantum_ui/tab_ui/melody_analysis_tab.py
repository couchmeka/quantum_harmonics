import numpy as np
from scipy.linalg import eigh
from scipy.integrate import odeint
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTextEdit,
    QGroupBox,
    QHBoxLayout,
    QComboBox,
    QScrollArea,
    QDialog,
)
import qtawesome as qta
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data.frequencies import frequency_systems
from data.elements import atomic_frequencies
from storage.data_manager import QuantumDataManager


class MelodyAnalysisTab(QWidget):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.system_combo = None
        self.more_info_btn = None
        self.data_manager = QuantumDataManager()

        # Physical constants
        self.hbar = 1.0545718e-34  # Reduced Planck constant
        self.viscosity = 1.81e-5  # Air viscosity
        self.density = 1.225  # Air density
        self.speed_of_sound = 343  # m/s

        # Data references
        self.frequency_systems = frequency_systems
        self.atomic_frequencies = atomic_frequencies

        # Initialize matplotlib figure and canvas
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)

        # Initialize UI components
        self.note_input = None
        self.results_text = None
        self.analyze_button = None

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Input section
        input_group = QGroupBox("Input Harmonies")
        input_group.setStyleSheet(
            """
            QGroupBox {
                color: white;
                font-size: 14px;
                border: 1px solid #3d405e;
                border-radius: 5px;
                padding: 15px;
                background-color: rgba(61, 64, 94, 0.3);
            }
        """
        )

        # Single row layout for all controls
        input_layout = QHBoxLayout()

        # Info button
        self.more_info_btn = QPushButton()
        self.more_info_btn.setIcon(qta.icon("fa.question-circle", color="#2196F3"))
        self.more_info_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                border: none;
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
            }
            QPushButton:hover {
                background-color: rgba(33, 150, 243, 0.1);
                border-radius: 16px;
            }
        """
        )
        self.more_info_btn.setToolTip("More information about the plots")
        self.more_info_btn.clicked.connect(self.show_plot_info)
        input_layout.addWidget(self.more_info_btn)

        # Note input label and field
        input_label = QLabel("Enter Notes:")
        input_label.setStyleSheet("color: white;")
        input_layout.addWidget(input_label)

        self.note_input = QLineEdit()
        self.note_input.setStyleSheet(
            """
            QLineEdit {
                background-color: white;
                padding: 5px;
                border-radius: 3px;
            }
        """
        )
        input_layout.addWidget(self.note_input)

        # System selector
        system_label = QLabel("Type:")
        system_label.setStyleSheet("color: white;")
        input_layout.addWidget(system_label)

        self.system_combo = QComboBox()
        self.system_combo.addItems(
            ["Western 12-Tone", "Indian Classical", "Arabic", "Gamelan", "Pythagorean"]
        )
        input_layout.addWidget(self.system_combo)

        # Analyze button
        self.analyze_button = QPushButton("Analyze Harmony")
        self.analyze_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(89, 92, 120, 0.6);
                color: white;
                border: 1px solid #3d405e;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(25, 118, 210, 0.3);
                border: 1px solid #1976D2;
            }
            QPushButton:pressed {
                background-color: rgba(13, 71, 161, 0.3);
            }
        """
        )
        self.analyze_button.clicked.connect(self.analyze_harmony)
        input_layout.addWidget(self.analyze_button)

        # Set the input group layout
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Create visualization container with horizontal layout
        viz_container = QWidget()
        viz_layout = QHBoxLayout(viz_container)

        # Left side: Canvas container
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        self.canvas.setMinimumHeight(400)
        canvas_layout.addWidget(self.canvas)

        # Right side: Results container
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)

        results_label = QLabel("Analysis Results")
        results_label.setStyleSheet("color: white; font-weight: bold;")
        results_layout.addWidget(results_label)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(
            """
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                padding: 8px;
                color: black;
                font-family: Arial;
                min-width: 300px;
                max-width: 400px;
            }
        """
        )
        results_layout.addWidget(self.results_text)

        # Add canvas and results to visualization container
        viz_layout.addWidget(
            canvas_container, stretch=2
        )  # Canvas takes 2/3 of the width
        viz_layout.addWidget(
            results_container, stretch=1
        )  # Results takes 1/3 of the width

        # Add visualization section to main layout
        layout.addWidget(viz_container)

        # Connect the system combo box to update note placeholder
        self.system_combo.currentTextChanged.connect(self.update_note_placeholder)

        # Initial update of note placeholder
        self.update_note_placeholder(self.system_combo.currentText())

    def create_hamiltonian(self, frequencies):
        """Create Hamiltonian matrix for the system"""
        N = len(frequencies)
        H = np.zeros((N, N), dtype=complex)

        # Kinetic energy terms (diagonal)
        for i in range(N):
            H[i, i] = frequencies[i] * self.hbar

        # Potential energy terms (off-diagonal coupling)
        for i in range(N - 1):
            coupling = np.sqrt(frequencies[i] * frequencies[i + 1]) * self.hbar / 2
            H[i, i + 1] = coupling
            H[i + 1, i] = coupling.conjugate()

        return H

    def navier_stokes_sound(self, y, t, L):
        """Simplified Navier-Stokes equations for sound wave propagation"""
        velocity, pressure = y

        dvelocity = -1 / (self.density * L) * pressure
        dpressure = (
            -self.density * self.speed_of_sound**2 / L * velocity
            + self.viscosity / L**2 * velocity
        )

        return [dvelocity, dpressure]

    def analyze_harmony(self):  # In MelodyAnalysisTab
        try:
            input_notes = [note.strip() for note in self.note_input.text().split(",")]
            frequencies = []
            musical_systems = {}

            print("\nMelody Analysis Input:")
            print(f"Input notes: {input_notes}")

            # Check all frequency systems
            for system_name, system in self.frequency_systems.items():
                system_frequencies = []
                system_notes = []
                for note in input_notes:
                    if note in system:
                        system_frequencies.append(system[note])
                        system_notes.append(note)

                if system_frequencies:
                    musical_systems[system_name] = {
                        "notes": system_notes,
                        "frequencies": system_frequencies,
                    }

            # Combine all found frequencies
            all_frequencies = []
            for system in musical_systems.values():
                all_frequencies.extend(system["frequencies"])

            if not all_frequencies:
                self.results_text.setText("No valid notes found in any musical system")
                return

            print(f"Musical systems found: {list(musical_systems.keys())}")
            print(f"Total frequencies found: {len(all_frequencies)}")

            hamiltonian = self.create_hamiltonian(all_frequencies)
            eigenvalues, eigenvectors = eigh(hamiltonian)

            t = np.linspace(0, 1, 1000)
            L = self.speed_of_sound / min(all_frequencies)
            y0 = [0, max(all_frequencies)]
            wave_solution = odeint(self.navier_stokes_sound, y0, t, args=(L,))

            # Format results
            melody_results = dict(
                musical_systems=musical_systems,
                notes=input_notes,
                frequencies=all_frequencies,
                eigenvalues=eigenvalues.tolist(),
                wave_solution=wave_solution.tolist(),
                t=t.tolist(),
            )

            print("\nMelody Results Structure:")
            for key, value in melody_results.items():
                if isinstance(value, dict):
                    print(f"{key}: {list(value.keys())}")
                elif isinstance(value, list):
                    print(f"{key}: {len(value)} items")
                else:
                    print(f"{key}: {type(value)}")

            # Store in data manager
            self.data_manager.update_melody_results(melody_results)
            print("Melody results stored in data manager")

            # Emit results
            self.analysis_complete.emit(melody_results)
            print("Analysis complete signal emitted")

            # Update visualization and display
            self.plot_analysis(
                all_frequencies, eigenvalues, eigenvectors, t, wave_solution
            )
            self.display_results(input_notes, all_frequencies, eigenvalues)

        except Exception as e:
            print(f"\nError in melody analysis: {str(e)}")
            self.results_text.setText(f"Error: {str(e)}")

    def plot_analysis(self, frequencies, eigenvalues, eigenvectors, t, wave_solution):
        """Create visualizations"""
        self.figure.clear()

        # 1. Frequency-Element Resonance
        ax1 = self.figure.add_subplot(221)
        atomic_weights = list(self.atomic_frequencies.values())
        resonance_matrix = np.zeros((len(frequencies), len(atomic_weights)))

        for i, freq in enumerate(frequencies):
            for j, weight in enumerate(atomic_weights):
                resonance = abs(freq - weight * 440) / (freq + weight * 440)
                resonance_matrix[i, j] = resonance

        im1 = ax1.imshow(resonance_matrix, aspect="auto", cmap="viridis")
        ax1.set_title("Frequency-Element Resonance")
        self.figure.colorbar(im1, ax=ax1)

        # 2. Energy Level Diagram
        ax2 = self.figure.add_subplot(222)
        for i, E in enumerate(eigenvalues):
            ax2.plot([-0.5, 0.5], [E, E], "b-")
        ax2.set_title("Quantum Energy Levels")
        ax2.set_ylabel("Energy (J)")

        # 3. Wave Evolution
        ax3 = self.figure.add_subplot(223)
        ax3.plot(t, wave_solution[:, 0], "b-", label="Velocity")
        ax3.plot(t, wave_solution[:, 1], "r-", label="Pressure")
        ax3.set_title("Sound Wave Evolution")
        ax3.legend()

        # 4. Phase Space
        ax4 = self.figure.add_subplot(224)
        phase_space = np.zeros((len(frequencies), 2))
        for i in range(len(frequencies)):
            phase_space[i] = [np.real(eigenvectors[i, 0]), np.imag(eigenvectors[i, 0])]
        ax4.plot(phase_space[:, 0], phase_space[:, 1], "bo-")
        ax4.set_title("Phase Space Trajectory")

        self.figure.tight_layout()
        self.canvas.draw()

    def display_results(self, notes, frequencies, eigenvalues):
        """Display analysis results in text area"""
        self.results_text.clear()
        self.results_text.append("=== Harmony Analysis Results ===\n")

        # Show note frequencies
        self.results_text.append("Input Notes and Frequencies:")
        for note, freq in zip(notes, frequencies):
            self.results_text.append(f"{note}: {freq:.2f} Hz")

        # Show quantum energy levels
        self.results_text.append("\nQuantum Energy Levels:")
        for i, energy in enumerate(eigenvalues):
            self.results_text.append(f"Level {i + 1}: {energy:.2e} J")

        # Show resonating elements
        self.results_text.append("\nResonating Elements:")
        for i, freq in enumerate(frequencies):
            resonances = []
            for element, weight in self.atomic_frequencies.items():
                element_freq = weight * 440
                if abs(freq - element_freq) / (freq + element_freq) < 0.1:
                    resonances.append(element)
            if resonances:
                self.results_text.append(
                    f"{notes[i]} resonates with: {', '.join(resonances)}"
                )

    def show_plot_info(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Plot Information")
        layout = QVBoxLayout(dialog)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QLabel()
        content.setWordWrap(True)
        content.setOpenExternalLinks(True)
        content.setTextFormat(Qt.TextFormat.RichText)

        info_text = """
        <h3 style='text-align: center;'>Visualization Details</h3>
        <ol>
        <li><b>Velocity Components:</b> Evolution of velocity components over time.</li>
        <li><b>Quantum Pressure Field:</b> Pressure evolution over time.</li>
        <li><b>Velocity Magnitude:</b> Magnitude of velocity vs. tunneling probability (log scale).</li>
        <li><b>Frequency Comparison:</b> Musical, atomic, quantum, and Fibonacci frequencies (log scale).</li>
        <li><b>Terahertz Transitions:</b> 3D surface plot of Gaussian distribution for transition intensities.</li>
        <li><b>Quantum-Classical Resonance:</b> Resonance strength across frequency range.</li>
        </ol>
        <p><b>Summary:</b> These visualizations bridge classical and quantum fluid dynamics, 
        revealing connections between musical harmonies and quantum phenomena. They illustrate 
        velocity evolution, pressure fields, energy spectra, quantum tunneling effects, 
        resonance patterns, and terahertz transitions, offering insights into quantum-classical 
        correspondences and applications in fluid dynamics and quantum computing.</p>
        """

        content.setText(info_text)
        scroll.setWidget(content)

        layout.addWidget(scroll)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setStyleSheet(
            """
            QDialog {
                min-width: 400px;
                max-width: 600px;
            }
            QLabel {
                margin: 10px;
            }
            QPushButton {
                min-width: 50px;
                min-height: 50px;
                padding: 5px 15px;
                margin: 10px auto;
            }
        """
        )

        dialog.exec()

    def update_note_placeholder(self, system):
        if system == "Western 12-Tone":
            self.note_input.setPlaceholderText("e.g., C4,E4,G4")  # Correct
        elif system == "Indian Classical":
            self.note_input.setPlaceholderText(
                "e.g., Sa,Re,Ga"
            )  # Changed from Sa,Ma,Pa to match the green section
        elif system == "Arabic":
            self.note_input.setPlaceholderText(
                "e.g., Duka,Sika,Jaharka"
            )  # Changed to match the red section
        elif system == "Gamelan":
            self.note_input.setPlaceholderText(
                "e.g., Slendro 1,2,3"
            )  # Added "Slendro" to be more specific
        elif system == "Pythagorean":
            self.note_input.setPlaceholderText("e.g., C4,F4,G4")  # Correct
