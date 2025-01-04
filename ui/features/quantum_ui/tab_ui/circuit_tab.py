from qiskit import qasm3

from core.calculations.quantum.quantum_circuit_builder import create_circuit
from core.calculations.quantum.quantum_harmonic_analysis import QuantumHarmonicsAnalyzer
from core.calculations.quantum.quantum_state_display import QuantumStateVisualizer
from data.frequencies import frequency_systems
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QComboBox,
)
from PyQt6.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from storage.data_manager import QuantumDataManager
from ui.styles_ui.styles import (
    create_group_box,
    DEFAULT_SPACING,
    DEFAULT_MARGINS,
    base_style,
    create_button,
    textedit_style,
)


class QuantumMelodyAnalysisTab(QWidget):
    analysis_complete = pyqtSignal(dict)  # Signal for when analysis is complete

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sample_rate = None
        self.audio_data = None
        self.frequency_systems = frequency_systems
        self.data_manager = QuantumDataManager()
        self.system_combo = None
        self.analyzer = QuantumHarmonicsAnalyzer()

        # Initialize visualization components
        self.main_figure = Figure(figsize=(12, 8))
        self.main_canvas = FigureCanvas(self.main_figure)
        self.state_figure = Figure(figsize=(8, 6))
        self.state_canvas = FigureCanvas(self.state_figure)

        self.visualizer = QuantumStateVisualizer(self.state_figure)
        self.last_results = None

        # UI Components
        self.note_input = None
        self.results_text = None

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(*DEFAULT_MARGINS)
        layout.setSpacing(DEFAULT_SPACING)

        # Input controls group
        input_group = create_group_box("Melody Input")
        input_layout = QVBoxLayout()

        # # Note input
        # note_layout = QHBoxLayout()
        # note_label = QLabel("Enter notes (comma-separated, e.g. C4,E4,G4):")
        # note_label.setStyleSheet(base_style)
        # self.note_input = QLineEdit()
        # self.note_input.setStyleSheet(lineedit_style)
        # note_layout.addWidget(note_label)
        # note_layout.addWidget(self.note_input)
        # input_layout.addLayout(note_layout)

        # Note input with combo box
        note_layout = QHBoxLayout()

        # Label for notes
        note_label = QLabel("Enter notes:")
        note_label.setStyleSheet(base_style)
        note_layout.addWidget(note_label)

        # Combo box for note systems
        self.system_combo = QComboBox()
        self.system_combo.addItems(
            ["Western 12-Tone", "Indian Classical", "Arabic", "Gamelan", "Pythagorean"]
        )
        note_layout.addWidget(self.system_combo)

        # Line edit for notes (with black text)
        self.note_input = QLineEdit()
        self.note_input.setStyleSheet(
            "color: black; background-color: white; border: 1px solid gray;"
        )  # Black text
        note_layout.addWidget(self.note_input)

        # Add the layout to the parent input layout
        input_layout.addLayout(note_layout)

        # Analysis and Export buttons
        button_layout = QHBoxLayout()
        analyze_btn = create_button("Run Quantum Analysis")
        analyze_btn.clicked.connect(self.run_analysis)
        button_layout.addWidget(analyze_btn)

        # Export to simulator buttons
        export_sim_btn = create_button("Export to Simulation")
        export_sim_btn.clicked.connect(self.export_to_simulation)
        # Disabled until analysis is complete
        button_layout.addWidget(export_sim_btn)

        input_layout.addLayout(button_layout)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Create visualization layout
        vis_layout = QHBoxLayout()

        # Main analysis visualization
        main_vis_group = create_group_box("Quantum Analysis")
        main_vis_layout = QVBoxLayout()
        main_vis_layout.addWidget(self.main_canvas)
        main_vis_group.setLayout(main_vis_layout)
        vis_layout.addWidget(main_vis_group)

        # Quantum state visualization
        state_vis_group = create_group_box("Quantum State")
        state_vis_layout = QVBoxLayout()
        state_vis_layout.addWidget(self.state_canvas)
        state_vis_group.setLayout(state_vis_layout)
        vis_layout.addWidget(state_vis_group)

        layout.addLayout(vis_layout)

        # Results text area
        results_group = create_group_box("Analysis Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(textedit_style)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.setLayout(layout)

    def run_analysis(self):
        try:
            notes = [note.strip() for note in self.note_input.text().split(",")]
            if not notes:
                self.results_text.setText("Please enter some notes")
                return

            frequencies = []
            amplitudes = []
            for note in notes:
                for system_name, system in frequency_systems.items():
                    if note in system:
                        frequencies.append(system[note])
                        amplitudes.append(1.0)
                        break

            if not frequencies:
                self.results_text.setText("No valid notes found")
                return

            # Create circuit and run simulation
            circuit, simulation_result = create_circuit(frequencies, amplitudes)

            circuit_data = {
                "qasm": qasm3.dumps(circuit),
                "num_qubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "frequencies": frequencies,
                "amplitudes": amplitudes,
                "counts": simulation_result.get_counts(),
                "simulation_status": simulation_result.status,
            }

            results = self.analyzer.analyze_harmonics(
                frequencies=frequencies, amplitudes=amplitudes
            )

            melody_data = {
                "notes": notes,
                "frequencies": frequencies,
                "amplitudes": amplitudes,
                "circuit_data": circuit_data,
                "analysis_results": results,
                "musical_systems": self._get_musical_systems(notes),
            }

            self.data_manager.update_melody_results(melody_data)
            self.last_results = melody_data
            self.analysis_complete.emit(melody_data)
            self.update_visualizations(results, circuit)
            self.display_results(results, melody_data["musical_systems"])

        except Exception as e:
            print(f"Error in melody analysis: {str(e)}")
            import traceback

            traceback.print_exc()
            self.results_text.setText(f"Error: {str(e)}")
        finally:
            print("Analysis completed")

    def update_visualizations(self, results, circuit):
        """Update the visualization with circuit and simulation results"""
        # Get circuit from results
        circuit = results["circuit"]

        # Clear previous visualization
        self.main_figure.clear()

        # Set up grid layout
        gs = self.main_figure.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)

        # Circuit diagram (top subplot)
        ax_circuit = self.main_figure.add_subplot(gs[0])
        circuit.draw(
            output="mpl",
            style={"backgroundcolor": "white"},
            initial_state=True,
            plot_barriers=True,
            justify="left",
            ax=ax_circuit,
        )
        ax_circuit.set_title("Quantum Circuit")

        # State probabilities (bottom subplot)
        ax_state = self.main_figure.add_subplot(gs[1])
        counts = results["counts"]
        # Convert counts to probabilities
        total_shots = sum(counts.values())
        probs = {state: count / total_shots for state, count in counts.items()}

        # Plot state probabilities
        states = list(probs.keys())
        probabilities = list(probs.values())
        ax_state.bar(range(len(states)), probabilities)
        ax_state.set_xticks(range(len(states)))
        ax_state.set_xticklabels(states, rotation=45)
        ax_state.set_title("State Probabilities")
        ax_state.set_ylabel("Probability")

        # Update canvas
        self.main_canvas.draw()

        # Store analysis results
        self.last_results = results

    def display_results(self, results, musical_systems):
        """Display analysis results"""
        output = ["=== Quantum Analysis Results ===\n", "Musical Systems Used:"]

        # Show which notes came from which systems
        for system, notes in musical_systems.items():
            output.append(f"{system}: {', '.join(notes)}")

        output.extend(
            [
                f"\nState Purity: {results['purity']:.3f}",
                f"Quantum Fidelity: {results['fidelity']:.3f}",
                "\nPythagorean Analysis:",
            ]
        )

        for result in results["pythagorean_analysis"]:
            output.append(f"Frequency {result['frequency']:.1f} Hz:")
            output.append(f"  - Closest interval: {result['closest_interval']}")
            output.append(f"  - Deviation: {result['deviation']:.3f}")
            output.append(f"  - Harmonic influence: {result['harmonic_influence']:.3f}")

        if "atomic_analysis" in results:
            output.append("\nAtomic Resonances:")
            for result in results["atomic_analysis"]:
                if "atomic_matches" in result:
                    output.append(f"\nFrequency {result['frequency']:.1f} Hz:")
                    for match in result["atomic_matches"]:
                        output.append(f"  - Element: {match['element']}")
                        output.append(f"  - Harmonic: {match['harmonic']}")

        self.results_text.setText("\n".join(output))

    def export_to_simulation(self):
        """Export analysis results to particle simulation"""
        if self.last_results:
            self.analysis_complete.emit(self.last_results)
            self.results_text.append("\nExported to particle simulation!")
        else:
            self.results_text.append("\nNo analysis results available to export.")

    def get_current_results(self):
        if hasattr(self, "last_results"):
            return self.last_results
        return None

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

    def _get_musical_systems(self, notes):
        """Identify which musical systems the notes belong to"""
        musical_systems = {}
        for system_name, system in self.frequency_systems.items():
            system_notes = [note for note in notes if note in system]
            if system_notes:
                musical_systems[system_name] = {
                    "notes": system_notes,
                    "frequencies": [system[note] for note in system_notes],
                }
        return musical_systems

    def update_audio_data(self, audio_data, sample_rate):
        """Update the audio data and sample rate"""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        # If you want to automatically analyze when new data comes in:
        if self.audio_data is not None:
            self.run_analysis()
