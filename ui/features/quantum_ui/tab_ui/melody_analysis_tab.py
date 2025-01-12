from datetime import datetime

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
    QHBoxLayout,
    QComboBox,
    QScrollArea,
    QDialog,
)
import qtawesome as qta
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data.universal_measurements.frequencies import frequency_systems
from data.universal_measurements.elements import atomic_frequencies
from data.backend_data_management.data_manager import QuantumDataManager
from ui.styles_ui.styles import (
    create_group_box,
    create_button,
    textedit_style,
    base_style,
)


class MelodyAnalysisTab(QWidget):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.circuit_figure = None
        self.resonance_figure = None
        self.resonance_canvas = None
        self.circuit_canvas = None
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

        # Initialize UI components
        self.note_input = None
        self.results_text = None
        self.analyze_button = None

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 5)
        layout.setSpacing(5)

        # Input controls group - matching Circuit tab style
        input_group = create_group_box("Melody Input")
        input_layout = QHBoxLayout()

        # Info button with blue icon
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
        self.more_info_btn.clicked.connect(self.show_plot_info)
        input_layout.addWidget(self.more_info_btn)

        # Note input section
        note_label = QLabel("Enter Notes:")
        note_label.setStyleSheet("color: white;")
        self.note_input = QLineEdit()
        self.note_input.setStyleSheet(
            """
            QLineEdit {
                background-color: white;
                padding: 5px;
                border-radius: 3px;
                min-width: 200px;
            }
        """
        )

        # Music system selector
        system_label = QLabel("Type:")
        system_label.setStyleSheet(base_style)

        self.system_combo = QComboBox()
        self.system_combo.addItems(
            ["Western 12-Tone", "Indian Classical", "Arabic", "Gamelan", "Pythagorean"]
        )
        # Analysis buttons
        analyze_btn = create_button("Run Quantum Analysis")
        analyze_btn.clicked.connect(self.run_analysis)
        input_layout.addWidget(analyze_btn)

        # Add all controls to input layout
        input_layout.addWidget(note_label)
        input_layout.addWidget(self.note_input)
        input_layout.addWidget(system_label)
        input_layout.addWidget(self.system_combo)
        input_group.setLayout(input_layout)
        input_layout.addWidget(analyze_btn)
        layout.addWidget(input_group)

        # Main content area with visualizations and results
        content_container = QWidget()
        content_layout = QHBoxLayout(content_container)

        # Left side: Visualizations in two panels
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)

        # Top visualization: Quantum Circuit
        circuit_group = create_group_box("Quantum State Energy Dynamics")
        circuit_layout = QVBoxLayout()
        self.circuit_figure = Figure(figsize=(10, 7))  # Create figure first
        self.circuit_canvas = FigureCanvas(self.circuit_figure)  # Then create canvas
        self.circuit_canvas.setMinimumHeight(200)
        circuit_layout.addWidget(self.circuit_canvas)
        circuit_group.setLayout(circuit_layout)
        viz_layout.addWidget(circuit_group)

        # Bottom visualization: Atomic Resonance
        resonance_group = create_group_box("Wave Dynamics and Resonance")
        resonance_layout = QVBoxLayout()
        self.resonance_figure = Figure(figsize=(10, 7))  # Create figure first
        self.resonance_canvas = FigureCanvas(
            self.resonance_figure
        )  # Then create canvas
        self.resonance_canvas.setMinimumHeight(200)
        resonance_layout.addWidget(self.resonance_canvas)
        resonance_group.setLayout(resonance_layout)
        viz_layout.addWidget(resonance_group)

        content_layout.addWidget(viz_container, stretch=5)
        viz_layout.addWidget(self.results_text, stretch=1)

        # Right side: Analysis Results
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        results_label = QLabel("Analysis Results")
        results_label.setStyleSheet("color: white; font-weight: bold;")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(
            textedit_style
            + """
            QTextEdit {
                min-width: 300px;
                max-width: 400px;
            }
        """
        )
        results_layout.addWidget(results_label)
        results_layout.addWidget(self.results_text)
        content_layout.addWidget(results_container, stretch=1)

        layout.addWidget(content_container)
        self.setLayout(layout)

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

    def run_analysis(self):  # In MelodyAnalysisTab
        try:
            input_notes = [note.strip() for note in self.note_input.text().split(",")]
            frequencies = []
            musical_systems = {}
            valid_notes = []
            all_frequencies = []

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
                        valid_notes.append(note)

                if system_frequencies:
                    musical_systems[system_name] = {
                        "notes": system_notes,
                        "frequencies": system_frequencies,
                    }
                all_frequencies.extend(system_frequencies)
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
                quantum_frequencies=all_frequencies,  # Renamed key
                eigenvalues=eigenvalues.tolist(),
                wave_solution=wave_solution.tolist(),
                t=t.tolist(),
                musical_systems=musical_systems,
                notes=input_notes,
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
                all_frequencies,
                eigenvalues,
                eigenvectors,
                t,
                wave_solution,
                valid_notes,
            )
            self.display_results(input_notes, all_frequencies, eigenvalues)

        except Exception as e:
            print(f"\nError in melody analysis: {str(e)}")
            self.results_text.setText(f"Error: {str(e)}")

    def plot_analysis(
        self, frequencies, eigenvalues, eigenvectors, t, wave_solution, input_notes
    ):
        """Create visualizations showing quantum-musical relationships"""
        # Clear both figures
        self.circuit_figure.clear()
        self.resonance_figure.clear()

        # Create subplots
        gs1 = self.circuit_figure.add_gridspec(1, 3)
        ax0 = self.circuit_figure.add_subplot(gs1[0])
        ax1 = self.circuit_figure.add_subplot(gs1[1])
        ax2 = self.circuit_figure.add_subplot(gs1[2])

        # 1. Energy Level Distribution
        normalized_eigenvalues = eigenvalues / max(abs(eigenvalues))
        unique_notes = []
        [unique_notes.append(x) for x in input_notes if x not in unique_notes]

        # Only plot for states with corresponding notes
        for i in range(len(unique_notes)):
            if i < len(normalized_eigenvalues):
                E = normalized_eigenvalues[i]
                ax0.plot([-0.5, 0.5], [E, E], "b-", alpha=0.5)
                prob = np.abs(eigenvectors[i][0]) ** 2
                ax0.scatter(0, E, s=prob * 500, color="red", alpha=0.6)
                ax0.annotate(
                    f"{unique_notes[i]} ({frequencies[i]:.1f} Hz)",
                    (-0.4, E),
                    va="center",
                )

        ax0.set_title("Energy Level Distribution")
        ax0.set_ylabel("Normalized Energy")
        ax0.set_xlabel("State")
        ax0.grid(True)

        # 2. Time Evolution of States
        for i in range(min(3, len(eigenvalues))):  # Show first 3 states
            evolution = np.exp(-1j * eigenvalues[i] * t)
            ax1.plot(t, np.real(evolution), label=f"State {i + 1}")
            print(f"Number of eigenvalues: {len(eigenvalues)}")

        ax1.set_title("Q State Evolution")
        ax1.set_ylabel("State Amplitude")
        ax1.set_xlabel("Time")
        ax1.legend(
            title="Quantum States",
            loc="upper right",
            bbox_to_anchor=(1.3, 1),
            fontsize="small",
        )

        ax1.grid(True)

        # 3. Musical-Quantum Correlation
        # Compare musical frequency ratios with eigenvalue ratios
        freq_ratios = np.array([f / min(frequencies) for f in frequencies])
        eigen_ratios = np.array(
            [e / min(eigenvalues) for e in eigenvalues[: len(frequencies)]]
        )

        ax2.scatter(freq_ratios, eigen_ratios, alpha=0.6)
        ax2.plot(
            [min(freq_ratios), max(freq_ratios)],
            [min(freq_ratios), max(freq_ratios)],
            "r--",
            label="Perfect Correlation",
        )
        ax2.set_title("Music-Quantum Correlation")
        ax2.set_xlabel("Frequency Ratio")
        ax2.set_ylabel("Energy Ratio")
        ax2.legend()
        ax2.grid(True)

        self.circuit_figure.tight_layout()
        self.circuit_canvas.draw()

        # Bottom resonance plots
        gs2 = self.resonance_figure.add_gridspec(1, 2)
        ax3 = self.resonance_figure.add_subplot(gs2[0])
        ax4 = self.resonance_figure.add_subplot(gs2[1])

        # 4. Wave Function Evolution
        velocity = wave_solution[:, 0]
        pressure = wave_solution[:, 1]
        ax3.plot(t, velocity, label="Velocity")
        ax3.plot(t, pressure, label="Pressure")

        # Add eigenvalue contributions
        for i, E in enumerate(eigenvalues[:2]):  # Show first 2 eigenvalues
            contribution = np.sin(E * t) * 0.5 * np.exp(-0.1 * t)
            ax3.plot(t, contribution, "--", label=f"E{i + 1} Contribution", alpha=0.5)

        ax3.set_title("Wave Evolution with Eigenvalue Contributions")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Amplitude")
        ax3.legend()
        ax3.grid(True)

        # 5. Phase Space with Eigenstates
        ax4.clear()  # Make sure the axis is clear
        for i in range(len(unique_notes)):
            if eigenvectors[i].size > 0 and i < len(unique_notes):
                real_part = np.real(eigenvectors[i][0])
                imag_part = np.imag(eigenvectors[i][0])
                size = 100 * (np.abs(eigenvalues[i]) / max(np.abs(eigenvalues)))
                ax4.scatter(
                    real_part, imag_part, s=size, alpha=0.6, label=f"State {i + 1}"
                )
                ax4.annotate(
                    unique_notes[i],
                    (real_part, imag_part),
                    xytext=(5, 5),
                    textcoords="offset points",
                )
        ax4.set_title("Phase Space Distribution")
        ax4.set_xlabel("Re(ψ)")
        ax4.set_ylabel("Im(ψ)")
        if len(frequencies) <= 5:  # Only show legend if we have few states
            ax4.legend()
        ax4.grid(True)

        self.resonance_figure.tight_layout()
        self.resonance_canvas.draw()

    def display_results(self, notes, frequencies, eigenvalues):
        self.results_text.clear()
        self.results_text.append("=== Quantum Musical Analysis ===\n")

        # Raw Data
        self.results_text.append("Input Data:")
        for note, freq in zip(notes, frequencies):
            self.results_text.append(f"{note}: {freq:.2f} Hz")

        # Plot Interpretations
        self.results_text.append("\n=== Visualization Guide ===")

        self.results_text.append("\nPhase Space Distribution:")
        self.results_text.append("• Shows quantum states of each note")
        self.results_text.append("• X-axis: Real component of state")
        self.results_text.append("• Y-axis: Imaginary component of state")
        self.results_text.append("• Point size: Energy level of state")
        self.results_text.append("• Clustering indicates harmonic relationships")
        self.results_text.append("• Labels show which note creates each state")

        self.results_text.append("\nEnergy Distribution:")
        self.results_text.append("• Shows energy levels of each note")
        self.results_text.append("• Horizontal lines: Possible energy states")
        self.results_text.append("• Red dots: Probability of each state")
        self.results_text.append("• Larger dots mean higher probability")
        self.results_text.append("• Labels show note and frequency")

        self.results_text.append("\nQuantum State Evolution:")
        self.results_text.append("• Shows how states change over time")
        self.results_text.append("• Each line represents one state")
        self.results_text.append("• Amplitude shows state strength")

        self.results_text.append("\nQuantum-Music Correlation:")
        self.results_text.append("• Compares frequency ratios to energy ratios")
        self.results_text.append("• Points near line = strong correlation")
        self.results_text.append("• Shows harmony-energy relationship")

        # Analysis Results
        self.results_text.append("\n=== Quantum Analysis ===")
        self.results_text.append("\nEnergy Levels:")
        for i, (note, energy, freq) in enumerate(zip(notes, eigenvalues, frequencies)):
            self.results_text.append(f"Note {note} ({freq:.1f} Hz):")
            self.results_text.append(f"• Energy: {energy:.2e} J")
            self.results_text.append(
                f"• Normalized Level: {energy / max(eigenvalues):.3f}"
            )

        # Harmonic Analysis
        self.results_text.append("\nHarmonic Relationships:")
        for i, note1 in enumerate(notes):
            for j, note2 in enumerate(notes[i + 1 :], i + 1):
                ratio = frequencies[j] / frequencies[i]
                self.results_text.append(f"{note1}-{note2}:")
                self.results_text.append(f"• Frequency Ratio: {ratio:.3f}")
                self.results_text.append(
                    f"• Energy Ratio: {eigenvalues[j] / eigenvalues[i]:.3f}"
                )

        # Store for ML
        melody_data = {
            "timestamp": datetime.now(),
            "data": {
                "notes": notes,
                "frequencies": frequencies,
                "eigenvalues": eigenvalues.tolist(),
                "musical_systems": {
                    "quantum_correlations": {
                        "notes": notes,
                        "frequencies": frequencies,
                    },
                },
                "wave_solution": None,
                "t": None,
                "quantum_frequencies": frequencies,
            },
        }
        self.data_manager.update_melody_results(melody_data)

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
        <li><b>Energy Level Distribution:</b> Shows the quantum energy levels (horizontal lines) 
        with red dots indicating the probability of finding the system in each state. Larger dots 
        mean higher probability.</li>

        <li><b>Q State Evolution:</b> Displays how the first three quantum states change 
        over time, showing the wave-like nature of musical quantum states.</li>

        <li><b>Music-Quantum Correlation:</b> Compares musical frequency ratios with quantum 
        energy ratios. Points closer to the red dashed line indicate stronger correlation between 
        musical and quantum properties.</li>

        <li><b>Wave Evolution with Eigenvalues:</b> Shows the classical wave behavior (velocity 
        and pressure) along with contributions from quantum eigenvalues, demonstrating how 
        quantum states influence the wave pattern.</li>

        <li><b>Phase Space Distribution:</b> Displays the quantum state distribution in phase 
        space, where point size indicates energy level and position shows the quantum state 
        components.</li>
        </ol>

        <p><b>Summary:</b> These visualizations show how musical notes create quantum states 
        and how these states evolve over time. They reveal the connection between musical 
        harmony and quantum mechanics, showing how musical intervals correspond to quantum 
        energy levels and wave patterns.</p>
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
