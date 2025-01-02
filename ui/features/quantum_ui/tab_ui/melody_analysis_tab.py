import numpy as np
from scipy.linalg import eigh
from scipy.integrate import odeint
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTextEdit,
    QGroupBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data.frequencies import frequency_systems
from data.elements import atomic_frequencies
from storage.data_manager import QuantumDataManager


class MelodyAnalysisTab(QWidget):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
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

        input_layout = QVBoxLayout()

        # Note input
        input_label = QLabel("Enter notes (comma-separated, e.g. C4,E4,G4):")
        input_label.setStyleSheet("color: white;")
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

        # Analyze button
        self.analyze_button = QPushButton("Analyze Harmony")
        self.analyze_button.setStyleSheet(
            """
            QPushButton {
                background-color: #3d405e;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """
        )
        self.analyze_button.clicked.connect(self.analyze_harmony)

        input_layout.addWidget(input_label)
        input_layout.addWidget(self.note_input)
        input_layout.addWidget(self.analyze_button)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Add visualization canvas
        layout.addWidget(self.canvas)

        # Results text area
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
            }
        """
        )
        layout.addWidget(self.results_text)

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
            melody_results = {
                "musical_systems": musical_systems,
                "notes": input_notes,
                "frequencies": all_frequencies,
                "eigenvalues": eigenvalues.tolist(),
                "wave_solution": wave_solution.tolist(),
                "t": t.tolist(),
            }

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
