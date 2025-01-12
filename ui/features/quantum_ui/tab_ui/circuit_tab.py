import traceback

import pandas as pd
from qiskit import qasm3
from core.calculations.melody.melody_atomic_mapping import AtomicResonanceAnalyzer
from core.calculations.quantum.quantum_circuit_builder import create_circuit
from core.calculations.melody.melody_arc_visualizer import create_arc_diagram
from core.calculations.quantum.quantum_harmonic_analysis import QuantumHarmonicsAnalyzer
from data.universal_measurements.elements import atomic_frequencies
from data.universal_measurements.frequencies import frequency_systems
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


from data.backend_data_management.data_manager import QuantumDataManager
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
        self.arc_canvas = None
        self.historical_runs = []
        self.more_info_btn = None
        self.sample_rate = None
        self.audio_data = None
        self.system_combo = None
        self.frequency_systems = frequency_systems
        self.analyzer = QuantumHarmonicsAnalyzer()
        self.atomic_analyzer = AtomicResonanceAnalyzer()
        self.data_manager = QuantumDataManager()

        # Initialize visualization components
        self.main_figure = Figure(figsize=(12, 8))
        self.main_canvas = FigureCanvas(self.main_figure)

        # Initialize arc diagram components
        self.arc_figure = Figure(figsize=(12, 6))  # Add this line
        self.arc_canvas = FigureCanvas(self.arc_figure)  # And this line

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

        # Music system selector
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

        # Arc diagram visualization
        arc_vis_group = create_group_box("Atomic Resonance Network")
        arc_vis_layout = QVBoxLayout()
        self.arc_canvas.setMinimumHeight(300)  # Increased height
        self.arc_canvas.setMaximumHeight(400)  # Increased max height
        arc_vis_layout.addWidget(self.arc_canvas)
        arc_vis_group.setLayout(arc_vis_layout)
        viz_layout.addWidget(arc_vis_group)

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

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(
            textedit_style
            + """
            min-width: 300px;
            max-width: 400px;
        """
        )
        results_layout.addWidget(results_label)
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

            # Standardize data format for other components
            quantum_results = {
                "quantum_frequencies": frequencies,
                "amplitudes": amplitudes,
                "circuit_data": circuit_data,
                "analysis_results": results,
                "statevector": results.get("statevector", []),
                "purity": results.get("purity", 0.0),
                "fidelity": results.get("fidelity", 0.0),
            }

            melody_data = {
                "notes": notes,
                "frequencies": frequencies,
                "amplitudes": amplitudes,
                "circuit_data": circuit_data,
                "analysis_results": results,
                "musical_systems": self._get_musical_systems(notes),
            }

            self.data_manager.update_melody_results(melody_data)
            self.last_results = quantum_results
            self.analysis_complete.emit(quantum_results)
            self.update_visualizations(results, circuit)
            self.display_results(results, melody_data["musical_systems"])

        except Exception as e:
            print(f"Error in melody analysis: {str(e)}")
            traceback.print_exc()

    def update_visualizations(self, results, circuit):
        """Update visualizations with quantum circuit and arc diagram"""
        # Clear figures
        self.main_figure.clear()

        # Draw quantum circuit
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

        # Extract data from atomic_analysis
        atomic_analysis = results.get("atomic_analysis", [])
        if not atomic_analysis:
            print("No atomic analysis found")
            return

        # Extract notes and frequencies
        notes = []
        frequencies = []

        for analysis in atomic_analysis:
            freq = analysis.get("frequency")
            if freq:
                frequencies.append(freq)
                # Get note from selected musical system
                musical_mappings = analysis.get("musical_mappings", {})
                # Try each system in order, use the first one that has a note
                note = None
                for system in [
                    "western_12_tone",
                    "indian_classical",
                    "arabic",
                    "gamelan",
                    "pythagorean",
                ]:
                    system_data = musical_mappings.get(system, {})
                    if isinstance(system_data, dict) and "note" in system_data:
                        note = system_data["note"]
                        break

                if note is None:
                    note = "Unknown"  # Fallback if no note is found in any system
                notes.append(note)

        print(f"Debug - Extracted notes: {notes}")
        print(f"Debug - Extracted frequencies: {frequencies}")

        if not notes or not frequencies:
            print("No valid notes or frequencies found")
            return

        # Get atomic resonance data
        atomic_results = self.atomic_analyzer.analyze_atomic_resonances(frequencies)

        # Create data for arc diagram
        relationships_data = []
        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                ratio = min(frequencies[i], frequencies[j]) / max(
                    frequencies[i], frequencies[j]
                )

                # Get element matches for both frequencies
                elements_i = (
                    atomic_results[i]["elements"] if i < len(atomic_results) else []
                )
                elements_j = (
                    atomic_results[j]["elements"] if j < len(atomic_results) else []
                )

                # Find common elements
                elements_i_set = set(match["element"] for match in elements_i)
                elements_j_set = set(match["element"] for match in elements_j)
                common_elements = elements_i_set.intersection(elements_j_set)

                # Get element information for source and target nodes
                source_elements = [
                    f"{e['element']}: {e['transition']}" for e in elements_i[:2]
                ]
                target_elements = [
                    f"{e['element']}: {e['transition']}" for e in elements_j[:2]
                ]

                # Create node labels with element information
                source_label = f"{notes[i]}\n{frequencies[i]:.1f} Hz\n" + "\n".join(
                    source_elements
                )
                target_label = f"{notes[j]}\n{frequencies[j]:.1f} Hz\n" + "\n".join(
                    target_elements
                )

                relationships_data.append(
                    {
                        "source": source_label,  # Use the new source_label here
                        "target": target_label,  # Use the new target_label here
                        "weight": ratio,
                        "common_elements": list(common_elements),
                    }
                )

                df = pd.DataFrame(relationships_data)

        # Get all unique nodes and their element information
        node_info = {}
        for i, note in enumerate(notes):
            elements = atomic_results[i]["elements"] if i < len(atomic_results) else []
            element_text = "\n".join(
                [f"{e['element']}: {e['transition']}" for e in elements[:2]]
            )
            node_info[f"{note}\n{frequencies[i]:.1f} Hz"] = element_text

        # Add node_info to the visualization call
        # In update_visualizations, replace the final create_arc_diagram call with:

        # Create and draw arc diagram
        self.arc_figure.clear()
        create_arc_diagram(
            df,
            source_col="source",
            target_col="target",
            weight_col="weight",
            bg_color="#f5e0c4",
            cmap="inferno",
            fig=self.arc_canvas.figure,
        )
        self.arc_canvas.draw()

        # Store results for history
        self.last_results = results

    # uses planks constant
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
