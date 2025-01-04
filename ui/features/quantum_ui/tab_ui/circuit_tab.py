import numpy as np
from qiskit import qasm3

from core.calculations.quantum.quantum_circuit_builder import create_circuit
from core.calculations.quantum.quantum_harmonic_analysis import QuantumHarmonicsAnalyzer
from core.calculations.quantum.quantum_state_display import QuantumStateVisualizer
from data.elements import atomic_frequencies
from data.frequencies import frequency_systems
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QPushButton,
    QDialog,
    QScrollArea,
)
import qtawesome as qta
from PyQt6.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from storage.data_manager import QuantumDataManager
from ui.styles_ui.styles import (
    create_group_box,
    DEFAULT_SPACING,
    DEFAULT_MARGINS,
    base_style,
    create_button,
    textedit_style,
    lineedit_style,
)


class QuantumMelodyAnalysisTab(QWidget):
    analysis_complete = pyqtSignal(dict)  # Signal for when analysis is complete

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = None
        self.historical_runs = []
        self.more_info_btn = None
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
        input_layout = QHBoxLayout()  # Changed to horizontal layout

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
        self.more_info_btn.setToolTip("More information about the analysis")
        self.more_info_btn.clicked.connect(self.show_plot_info)
        input_layout.addWidget(self.more_info_btn)

        # Note input
        note_label = QLabel("Enter Notes:")
        note_label.setStyleSheet(base_style)
        input_layout.addWidget(note_label)

        self.note_input = QLineEdit()
        self.note_input.setStyleSheet(lineedit_style)
        input_layout.addWidget(self.note_input)

        # System selector
        system_label = QLabel("Type:")
        system_label.setStyleSheet(base_style)
        input_layout.addWidget(system_label)

        self.system_combo = QComboBox()
        self.system_combo.addItems(
            ["Western 12-Tone", "Indian Classical", "Arabic", "Gamelan", "Pythagorean"]
        )
        input_layout.addWidget(self.system_combo)

        # Analysis button
        analyze_btn = create_button("Run Quantum Analysis")
        analyze_btn.clicked.connect(self.run_analysis)
        input_layout.addWidget(analyze_btn)

        # Export button
        export_sim_btn = create_button("Export to Simulation")
        export_sim_btn.clicked.connect(self.export_to_simulation)
        input_layout.addWidget(export_sim_btn)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Create main content area with visualizations and results
        content_container = QWidget()
        content_layout = QHBoxLayout(content_container)

        # In the setup_ui method, update the visualization section:

        # Left side: Visualizations
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setSpacing(20)  # Space between visualizations

        # Main quantum circuit visualization
        main_vis_group = create_group_box("Quantum Circuit Analysis")
        main_vis_layout = QVBoxLayout()
        self.main_canvas.setMinimumHeight(250)
        self.main_canvas.setMaximumHeight(300)
        main_vis_layout.addWidget(self.main_canvas)
        main_vis_group.setLayout(main_vis_layout)
        viz_layout.addWidget(main_vis_group)

        # Note relationships with atomic data
        resonance_vis_group = create_group_box("Note Relationships")
        resonance_vis_layout = QVBoxLayout()
        self.state_canvas.setMinimumHeight(250)
        self.state_canvas.setMaximumHeight(300)
        resonance_vis_layout.addWidget(self.state_canvas)
        resonance_vis_group.setLayout(resonance_vis_layout)
        viz_layout.addWidget(resonance_vis_group)

        # Set stretch factors for visualizations
        viz_layout.setStretch(0, 1)  # Equal space for both
        viz_layout.setStretch(1, 1)

        content_layout.addWidget(
            viz_container, stretch=2
        )  # Visualization gets 2/3 of space

        # Right side: Results
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)

        results_label = QLabel("Analysis Results")
        results_label.setStyleSheet("color: white; font-weight: bold;")
        results_layout.addWidget(results_label)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(
            textedit_style
            + """
            min-width: 300px;
            max-width: 400px;
        """
        )
        results_layout.addWidget(self.results_text)

        content_layout.addWidget(results_container, stretch=1)

        # Add the content container to main layout
        layout.addWidget(content_container)

        self.setLayout(layout)

        # Connect the system combo box to update note placeholder
        self.system_combo.currentTextChanged.connect(self.update_note_placeholder)

        # Initial update of note placeholder
        self.update_note_placeholder(self.system_combo.currentText())

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
        """Update visualizations with proper data extraction"""
        # Clear figures
        self.main_figure.clear()
        self.state_figure.clear()

        # Draw circuit
        ax_circuit = self.main_figure.add_subplot(111)
        self.main_figure.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.2)
        circuit.draw(
            output="mpl",
            style={
                "backgroundcolor": "#f8f9fa",
                "fontsize": 10,
                "compress": True,
                "scale": 0.55,
            },
            initial_state=True,
            plot_barriers=True,
            justify="center",
            ax=ax_circuit,
        )
        self.main_canvas.draw()

        # Note relationships visualization
        ax_rel = self.state_figure.add_subplot(111)
        self.state_figure.set_size_inches(10, 5)

        # Extract data from atomic_analysis
        atomic_analysis = results.get("atomic_analysis", [])
        if not atomic_analysis:
            print("No atomic analysis found")
            return

        notes = []
        frequencies = []
        mappings = []

        for analysis in atomic_analysis:
            freq = analysis.get("frequency")
            if freq:
                frequencies.append(freq)
                # Get note from western_12_tone mapping
                note = (
                    analysis.get("musical_mappings", {})
                    .get("western_12_tone", {})
                    .get("note", "")
                )
                notes.append(note)
                mappings.append(analysis.get("atomic_matches", []))

        print(f"Debug - Extracted notes: {notes}")
        print(f"Debug - Extracted frequencies: {frequencies}")

        if not notes or not frequencies:
            print("No valid notes or frequencies found")
            return

        # Calculate positions
        radius = 1.0
        angles = np.linspace(0, 2 * np.pi, len(notes), endpoint=False)
        angles = angles + np.pi / 2  # Rotate to start from top
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)

        # Draw connections
        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                ratio = min(frequencies[i], frequencies[j]) / max(
                    frequencies[i], frequencies[j]
                )

                # Calculate curve points
                mid_x = (x[i] + x[j]) / 2
                mid_y = (y[i] + y[j]) / 2
                ctrl_x = mid_x + (y[j] - y[i]) * 0.2
                ctrl_y = mid_y - (x[j] - x[i]) * 0.2

                vertices = np.array(
                    [[x[i], y[i]], [ctrl_x, ctrl_y], [x[j], y[j]]], dtype=np.float64
                )

                path = Path(vertices, [Path.MOVETO, Path.CURVE3, Path.CURVE3])

                # Draw connection line
                patch = PathPatch(
                    path,
                    facecolor="none",
                    edgecolor="#00FFFF",
                    linewidth=2.5,
                    alpha=0.8,
                    zorder=2,
                )
                ax_rel.add_patch(patch)

                # Add ratio label
                ax_rel.annotate(
                    f"{ratio:.3f}",
                    (ctrl_x, ctrl_y),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1),
                    fontsize=8,
                    zorder=3,
                )

            # Draw nodes and labels with atomic information
        for i, (note, freq, atomic_matches) in enumerate(
            zip(notes, frequencies, mappings)
        ):
            # Map frequency to elements
            element_matches = self.map_frequency_to_elements(freq)
            if element_matches:
                # Element text box with note included
                elements = [match["element"] for match in element_matches]
                transitions = [match["transition"] for match in element_matches]

                # Include note in the element box
                element_text = (
                    f"Note: {note}\nElements: {', '.join(elements)}\n{transitions[0]}"
                )

                # Position element info
                text_angle = angles[i]
                text_radius = radius * 1.3
                text_x = text_radius * np.cos(text_angle)
                text_y = text_radius * np.sin(text_angle)

                # Create more compact box
                box = ax_rel.annotate(
                    element_text,
                    (text_x, text_y),
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="black",
                        alpha=1.0,
                        boxstyle="round,pad=0.5",
                    ),
                    fontsize=8,
                    zorder=5,
                )

                # Draw frequency connection line from node to box
                connector = PathPatch(
                    Path([(x[i], y[i]), (text_x, text_y)]),
                    facecolor="none",
                    edgecolor="#3d405e",
                    linestyle="--",
                    alpha=0.3,
                    zorder=1,
                )
                ax_rel.add_patch(connector)

            # Draw node (smaller now since we have better labels)
            ax_rel.scatter(
                x[i],
                y[i],
                s=200,
                color="white",
                edgecolor="#3d405e",
                linewidth=2,
                zorder=4,
            )

            # Just frequency at node
            ax_rel.annotate(
                f"{freq:.1f} Hz",
                (x[i], y[i]),
                ha="center",
                va="center",
                fontsize=8,
                zorder=5,
            )

            # Add hover event handling

        # Rest of visualization code remains the same (aspect, limits, etc.)
        ax_rel.set_aspect("equal")
        ax_rel.axis("off")
        ax_rel.set_xlim(-2, 2)
        ax_rel.set_ylim(-2, 2)

        self.state_figure.tight_layout(pad=1.5)
        self.state_canvas.draw()

        # Store results for history
        self.last_results = results

    def map_frequency_to_elements(self, frequency):
        """Map a frequency to possible atomic elements"""
        matches = []
        scaling_factor = 1 / 4.136  # Planck constant scaling
        target_mass = frequency * scaling_factor

        for element, atomic_mass in atomic_frequencies.items():
            # Check harmonics 1-4
            for harmonic in range(1, 5):
                harmonic_mass = atomic_mass * harmonic
                deviation = abs(harmonic_mass - target_mass) / harmonic_mass
                if deviation < 0.05:  # 5% tolerance
                    matches.append(
                        {
                            "element": element,
                            "harmonic": harmonic,
                            "deviation": deviation,
                            "transition": f"{harmonic}→{harmonic + 1}",
                        }
                    )

        return matches

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

    def update_audio_data(self, audio_data, sample_rate):
        """Update the audio data and sample rate"""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        # If you want to automatically analyze when new data comes in:
        if self.audio_data is not None:
            self.run_analysis()
