# Library Imports
import traceback
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QComboBox,
    QLineEdit,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal

# App Imports
from core.calculations.ai.ai_quantum_interpreter import QuantumInterpreter
from core.calculations.ai.quantum_results_interpreter import QuantumResultsInterpreter
from core.calculations.universal_theory.universal_theory_analysis import (
    UnifiedHarmonicSystem,
)
from core.calculations.ai.ai_unified_analysis import UnifiedAIAnalyzer
from core.calculations.spectroscopy.spectroscopy_analyzer import SpectroscopyAnalyzer

# Style imports
from ui.styles_ui.styles import (
    textedit_style,
    create_group_box,
    lineedit_style,
    create_button,
    DEFAULT_SPACING,
    DEFAULT_MARGINS,
    base_style,
)

# Data Manager **** Do not delete
from data.backend_data_management.data_manager import QuantumDataManager


class AIAnalysisScreen(QWidget):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, data_manager=None):
        super().__init__()
        if not data_manager:
            raise ValueError("data_manager is required")
        self.user_input = None
        self.data_manager = data_manager or QuantumDataManager()
        self.interpreter = QuantumResultsInterpreter()
        self.quantum_interpreter = QuantumInterpreter()
        self.unified_analyzer = UnifiedAIAnalyzer()
        self.spectroscopy = SpectroscopyAnalyzer()
        self.unified_system = UnifiedHarmonicSystem()

        # Initialize UI components
        self.figure = Figure(figsize=(15, 8))
        self.canvas = FigureCanvas(self.figure)
        self.analysis_text = None
        self.data_source_combo = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(*DEFAULT_MARGINS)
        layout.setSpacing(DEFAULT_SPACING)

        # Header controls group
        controls_group = create_group_box("AI Analysis Controls")
        controls_layout = QHBoxLayout()

        data_source_label = QLabel("Select Data Source:")
        data_source_label.setStyleSheet(base_style)
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(
            [
                "All Results",
                "Quantum Analysis",
                "Melody Analysis",
                "Fluid Dynamics",
                "QEC Analysis",
                "Particle Simulation",
            ]
        )

        analyze_btn = create_button("Run AI Analysis")
        analyze_btn.clicked.connect(self.analyze_results)

        controls_layout.addWidget(data_source_label)
        controls_layout.addWidget(self.data_source_combo)
        controls_layout.addWidget(analyze_btn)
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Main content area
        content_container = QWidget()
        content_layout = QHBoxLayout(content_container)

        # Left: Visualizations
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)

        # # Initialize matplotlib figure
        # self.figure = Figure(figsize=(15, 12))
        # self.canvas = FigureCanvas(self.figure)

        viz_group = create_group_box("AI Analysis Visualizations")
        viz_inner_layout = QVBoxLayout()
        self.canvas.setMinimumHeight(500)
        viz_inner_layout.addWidget(self.canvas)
        viz_group.setLayout(viz_inner_layout)
        viz_layout.addWidget(viz_group)
        content_layout.addWidget(viz_container, stretch=2)

        # Right: Analysis & Input
        analysis_container = QWidget()
        analysis_layout = QVBoxLayout(analysis_container)

        # Initialize QTextEdit for analysis
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setStyleSheet(
            textedit_style
            + """
            min-width: 300px;
            max-width: 400px;
            """
        )

        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Ask a question about the quantum data...")
        self.user_input.setStyleSheet(lineedit_style)

        ask_button = create_button("Ask AIQuantum")
        ask_button.clicked.connect(self.ask_gpt)

        analysis_layout.addWidget(self.analysis_text)
        analysis_layout.addWidget(self.user_input)
        analysis_layout.addWidget(ask_button)

        content_layout.addWidget(analysis_container, stretch=1)
        layout.addWidget(content_container)

        self.setLayout(layout)

    def analyze_results(self):
        try:
            analysis_results = self.collect_and_analyze_data()
            pattern_analysis = self.data_manager.analyze_ml_patterns()
            analyzed_datapoints = self.data_manager.get_all_results()

            self.figure.clear()
            gs = self.figure.add_gridspec(
                2,
                2,
                height_ratios=[1, 1],  # Equal height for rows
                width_ratios=[1, 1],  # Equal width for columns
                hspace=0.3,  # Reduced vertical space between plots
                wspace=0.3,  # Space between columns
                top=0.95,  # Less padding at top
                bottom=0.1,  # Less padding at bottom
                left=0.1,  # Left margin
                right=0.9,  # Right margin
            )

            # Plot 1: System Interactions
            ax1 = self.figure.add_subplot(gs[0, 0])
            self._plot_system_interactions(ax1, analysis_results["data_series"])

            # Plot 2: QEC Impact per Frequency
            ax2 = self.figure.add_subplot(gs[0, 1])
            self._plot_qec_frequency_impact(ax2, analysis_results["data_series"])

            # Plot 3: Note-QEC Correlation
            ax3 = self.figure.add_subplot(gs[1, 0])
            self._plot_note_qec_correlation(ax3, analysis_results["data_series"])

            # Plot 4: Pattern Analysis Results
            ax4 = self.figure.add_subplot(gs[1, 1])
            self._plot_ml_patterns(ax4, pattern_analysis)

            # Adjust title padding for all subplots
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_title(ax.get_title(), pad=10)

            self.canvas.draw()
            self._update_analysis_text(
                analysis_results, pattern_analysis, analyzed_datapoints
            )

        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            traceback.print_exc()

    def _build_analysis_prompt(self, results, pattern_analysis, analyzed_datapoints):
        try:
            data_series = results.get("data_series", {})

            # Extract quantum data with validation
            quantum_data = data_series.get("quantum", [])
            latest_quantum = quantum_data[-1] if quantum_data else {}

            # Get statevector in proper format
            statevector = latest_quantum.get("statevector", [])
            if isinstance(statevector, list) and len(statevector) > 0:
                # Format complex numbers properly
                formatted_statevector = [
                    f"{x:.3f}" for x in statevector[:5]
                ]  # Show first 5 elements
                statevector_str = f"[{', '.join(formatted_statevector)}...]"
            else:
                statevector_str = "[]"

            # Get QEC data with material info
            qec_data = data_series.get("qec", [])
            qec_info = []
            for qec_entry in qec_data:
                if isinstance(qec_entry, dict):
                    metrics = qec_entry.get("metrics", {})
                    material = qec_entry.get("material", {})
                    material_name = material.get("name", "Unknown")  # Get material name
                    qec_info.append(
                        f"""
                    QEC Circuit Type: {qec_entry.get('qec_type', 'unknown')}
                    Material: {material_name}
                    Temperature: {material.get('temperature', 0)}K
                    Initial Fidelity: {metrics.get('initial_fidelity', 0):.3f}
                    Final Fidelity: {metrics.get('final_fidelity', 0):.3f}
                    Recovery Rate: {metrics.get('recovery_rate', 0):.3f}
                    Error Suppression: {metrics.get('error_suppression', 0):.3f}
                    """
                    )

            # Build the prompt with explicit data
            prompt = f"""
            Comprehensive Quantum Analysis Query:
            ----------------------------------

            Quantum System State:
            • State Vector (first 5 elements): {statevector_str}
            • Purity: {latest_quantum.get('purity', 0):.3f}
            • Fidelity: {latest_quantum.get('fidelity', 0):.3f}
            • Number of Frequencies: {len(latest_quantum.get('quantum_frequencies', []))}
            • Raw Frequencies: {latest_quantum.get('quantum_frequencies', [])[:5]} (first 5 shown)

            QEC Analysis:
            {chr(10).join(qec_info)}

            Musical Analysis:
            • Notes: {data_series.get('melody', [{}])[-1].get('notes', [])}
            • Musical Systems: {list(data_series.get('melody', [{}])[-1].get('musical_systems', {}).keys())}
            • Frequencies: {data_series.get('melody', [{}])[-1].get('quantum_frequencies', [])}

            Fluid Analysis:
            • Original Frequencies: {data_series.get('fluid', [{}])[-1].get('original_frequencies', [])}
            • Fibonacci Frequencies: {data_series.get('fluid', [{}])[-1].get('fibonacci_frequencies', [])}

            Pattern Analysis:
            • Clusters: {len(pattern_analysis.get('clusters', []))}
            • Features: {pattern_analysis.get('n_features', 0)}
            • Key Features: {', '.join(list(pattern_analysis.get('feature_importance', {}).keys())[:3])}
            • Explained Variance: {sum(pattern_analysis.get('explained_variance', [0])) * 100:.2f}%

            User Question: {self.user_input.text()}
            """

            print("Debug - Generated prompt:")  # Debug output
            print(prompt)

            return prompt

        except Exception as e:
            print(f"Error building analysis prompt: {str(e)}")
            traceback.print_exc()
            return ""

    def ask_gpt(self):
        """Process data and generate a prompt for the quantum interpreter"""
        try:
            # Collect current data with error checking
            results = self.collect_and_analyze_data()
            pattern_analysis = self.data_manager.analyze_ml_patterns()
            analyzed_datapoints = self.data_manager.get_all_results()

            if not results or not isinstance(results, dict):
                print("No valid analysis data available")
                return "Error: No valid analysis data available"

            # Build prompt and get interpretation
            prompt = self._build_analysis_prompt(
                results, pattern_analysis, analyzed_datapoints
            )
            interpretation = self.quantum_interpreter.interpret_quantum_data(
                {"prompt": prompt}
            )

            # Update UI
            if interpretation and self.analysis_text:
                current_text = self.analysis_text.toPlainText()
                self.analysis_text.setText(
                    f"{current_text}\n\nAI Response:\n{interpretation}"
                )

            return interpretation

        except Exception as e:
            error_msg = f"Error in ask_gpt: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            if hasattr(self, "analysis_text") and self.analysis_text:
                self.analysis_text.setText(f"Error processing request: {str(e)}")
            return error_msg

    def safe_get_latest(self, data_list):
        """Safely get the latest item from a list"""
        if isinstance(data_list, list) and data_list:
            return data_list[-1]
        return {}

    def get_filtered_results(self, selected_source):
        """Helper method to get filtered results based on selection"""
        try:
            all_results = self.data_manager.get_all_results()
            if selected_source == "All Results":
                return {k: v for k, v in all_results.items() if v is not None}

            source_map = {
                "Quantum Analysis": "quantum",
                "Circuit": "quantum",
                "Melody Analysis": "melody",
                "Fluid Dynamics": "fluid",
                "QEC Analysis": "qec",
                "Particle Simulation": "particle",  # Add this line
            }

            key = source_map.get(selected_source)
            if key and key in all_results and all_results[key]:
                return {key: all_results[key]}
            return {}

        except Exception as e:
            print(f"Error filtering results: {str(e)}")
            return {}

    def display_and_analyze_results(self, results):
        try:
            # Generate analysis text
            analysis_text = ["=== Comprehensive Quantum Systems Analysis ===\n"]

            # System Overview
            active_sources = []
            for source, data in results.items():
                if data:  # Check if the data exists
                    active_sources.append(f"{source} Analysis")

            if active_sources:
                analysis_text.append(
                    f"• Active Data Sources: {', '.join(active_sources)}"
                )
            else:
                analysis_text.append("• No active data sources found")

            # Quantum Analysis
            if "quantum" in results:
                quantum_data = results["quantum"]
                analysis_text.append("\nQuantum State Analysis:")
                if "purity" in quantum_data:
                    analysis_text.append(
                        f"• State Purity: {quantum_data['purity']:.3f}"
                    )
                if "fidelity" in quantum_data:
                    analysis_text.append(
                        f"• Quantum Fidelity: {quantum_data['fidelity']:.3f}"
                    )

            # QEC Analysis
            if "qec" in results:
                qec_data = results["qec"]
                analysis_text.append("\nQuantum Error Correction:")
                if "metrics" in qec_data:
                    metrics = qec_data["metrics"]
                    analysis_text.append(
                        f"• Initial Fidelity: {metrics.get('initial_fidelity', 0):.3f}"
                    )
                    analysis_text.append(
                        f"• Final Fidelity: {metrics.get('final_fidelity', 0):.3f}"
                    )

            # Melody Analysis
            if "melody" in results:
                melody_data = results["melody"]
                analysis_text.append("\nQuantum Melody Analysis:")
                if "musical_systems" in melody_data:
                    for system, data in melody_data["musical_systems"].items():
                        notes = data.get("notes", [])
                        analysis_text.append(f"• {system}: {len(notes)} notes")

            # Fluid Analysis
            if "fluid" in results:
                fluid_data = results["fluid"]
                analysis_text.append("\nFluid Dynamics Analysis:")
                if "original_frequencies" in fluid_data:
                    analysis_text.append(
                        f"• Original Frequencies: {len(fluid_data['original_frequencies'])}"
                    )
                if "fibonacci_frequencies" in fluid_data:
                    analysis_text.append(
                        f"• Fibonacci Frequencies: {len(fluid_data['fibonacci_frequencies'])}"
                    )

            # Update the text display
            self.analysis_text.setText("\n".join(analysis_text))

            # Update results to prompt
            self.ask_gpt(results)

            # Update visualizations
            if hasattr(self, "update_visualizations"):
                self.update_visualizations(results)

        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            traceback.print_exc()
            self.analysis_text.setText(f"Error performing analysis: {str(e)}")

    def emit_analysis_results(self, results):
        self.analysis_complete.emit(results)

    def _plot_correlations(self, ax, correlation_analysis):
        correlations = {
            "Q-M": correlation_analysis.get("quantum_melody_correlation", 0),
            "Q-F": correlation_analysis.get("quantum_fluid_correlation", 0),
            "M-F": correlation_analysis.get("melody_fluid_correlation", 0),
        }

        positions = range(len(correlations))
        ax.bar(positions, correlations.values(), color="purple")
        ax.set_title("Cross-System Correlations")
        ax.set_xticks(positions)
        ax.set_xticklabels(correlations.keys())
        ax.set_ylabel("Correlation Strength")

    def _plot_qec_effectiveness(self, ax, qec_data):
        if not qec_data:
            ax.text(0.5, 0.5, "No QEC data available", ha="center", va="center")
            return

        timepoints = []
        signals = []
        regions = []
        markers = {"basic": "o", "surface": "s", "color": "^"}

        for data in qec_data:
            metrics = data.get("metrics", {})
            timepoints.append(data.get("timestamp", 0))
            signals.append(metrics.get("fidelity", 0))
            regions.append(data.get("qec_type", "unknown"))

        # Plot different QEC types with different markers/styles
        for qec_type in set(regions):
            mask = [r == qec_type for r in regions]
            ax.plot(
                [t for t, m in zip(timepoints, mask) if m],
                [s for s, m in zip(signals, mask) if m],
                marker=markers.get(qec_type, "x"),
                label=qec_type,
                linestyle="-",
            )

        ax.set_title("QEC Performance Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Fidelity")
        ax.legend()

    def _plot_qec_frequency_impact(self, ax, data_series):
        """Plot QEC impact on different frequency ranges"""
        try:
            qec_data = data_series.get("qec", [])
            if not qec_data:
                ax.text(0.5, 0.5, "No QEC data available", ha="center", va="center")
                return

            frequencies = []
            impacts = []

            for entry in qec_data:
                if isinstance(entry, dict):
                    metrics = entry.get("metrics", {})
                    if metrics:
                        freq = metrics.get("frequency", 0)
                        impact = metrics.get("improvement", 0)
                        if freq and isinstance(impact, (int, float)):
                            frequencies.append(freq)
                            impacts.append(impact)

            if frequencies and impacts:
                ax.scatter(frequencies, impacts, alpha=0.6)
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("QEC Impact (%)")
                ax.set_title("QEC Performance vs Frequency")
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, "Insufficient QEC data", ha="center", va="center")

        except Exception as e:
            print(f"Error plotting QEC frequency impact: {str(e)}")
            ax.text(0.5, 0.5, "Error plotting data", ha="center", va="center")

    def _plot_note_qec_correlation(self, ax, data_series):
        """Plot correlation between musical notes and QEC performance"""
        try:
            melody_data = data_series.get("melody", [])
            qec_data = data_series.get("qec", [])

            if not melody_data or not qec_data:
                ax.text(
                    0.5, 0.5, "No correlation data available", ha="center", va="center"
                )
                return

            notes = []
            qec_effectiveness = []

            for m_entry in melody_data:
                if isinstance(m_entry, dict):
                    notes.extend(m_entry.get("notes", []))

            for q_entry in qec_data:
                if isinstance(q_entry, dict):
                    metrics = q_entry.get("metrics", {})
                    if metrics:
                        qec_effectiveness.append(metrics.get("improvement", 0))

            if notes and qec_effectiveness:
                # Ensure equal lengths for plotting
                min_len = min(len(notes), len(qec_effectiveness))
                ax.bar(notes[:min_len], qec_effectiveness[:min_len])
                ax.set_xlabel("Musical Notes")
                ax.set_ylabel("QEC Effectiveness (%)")
                ax.set_title("Note-QEC Correlation")
                plt.setp(ax.get_xticklabels(), rotation=45)
            else:
                ax.text(
                    0.5, 0.5, "Insufficient correlation data", ha="center", va="center"
                )

        except Exception as e:
            print(f"Error plotting note-QEC correlation: {str(e)}")
            ax.text(0.5, 0.5, "Error plotting correlation", ha="center", va="center")

    def _plot_ml_patterns(self, ax, pattern_analysis):
        """Plot ML pattern analysis results"""
        try:
            if not pattern_analysis or "pca_components" not in pattern_analysis:
                ax.text(0.5, 0.5, "No pattern data available", ha="center", va="center")
                return

            components = np.array(pattern_analysis["pca_components"])
            clusters = np.array(pattern_analysis["clusters"])

            if len(components) > 0 and len(components) == len(clusters):
                scatter = ax.scatter(
                    components[:, 0], components[:, 1], c=clusters, cmap="viridis"
                )
                ax.set_xlabel("First Principal Component")
                ax.set_ylabel("Second Principal Component")
                ax.set_title("Pattern Analysis")
                ax.legend(*scatter.legend_elements(), title="Clusters")
            else:
                ax.text(0.5, 0.5, "Insufficient pattern data", ha="center", va="center")

        except Exception as e:
            print(f"Error plotting ML patterns: {str(e)}")
            ax.text(0.5, 0.5, "Error plotting patterns", ha="center", va="center")

    def _plot_system_interactions(self, ax, data):
        """Plot interactions between quantum, QEC, and fluid systems"""
        try:
            systems = ["quantum", "qec", "melody", "fluid"]
            measurements = [len(data.get(sys, [])) for sys in systems]

            for i, (sys1, m1) in enumerate(zip(systems, measurements)):
                for j, (sys2, m2) in enumerate(
                    zip(systems[i + 1 :], measurements[i + 1 :])
                ):
                    if m1 > 0 and m2 > 0:
                        ax.plot([i, i + j + 1], [m1, m2], "-o", label=f"{sys1}-{sys2}")

            ax.set_xticks(range(len(systems)))
            ax.set_xticklabels(systems)
            ax.set_title("System Interactions")
            ax.legend()

        except Exception as e:
            print(f"Error plotting system interactions: {str(e)}")
            ax.text(0.5, 0.5, "Error plotting data", ha="center", va="center")

    def _plot_qec_frequency_impact(self, ax, data_series):
        """Plot QEC impact on different frequency ranges"""
        try:
            qec_data = data_series.get("qec", [])
            print(f"QEC Data found: {qec_data}")  # Debug print

            if not qec_data:
                ax.text(0.5, 0.5, "No QEC data available", ha="center", va="center")
                return

            frequencies = []
            fidelities = []
            improvements = []

            for entry in qec_data:
                if isinstance(entry, dict) and "metrics" in entry:
                    metrics = entry.get("metrics", {})
                    # Try to get frequency from quantum frequencies if available
                    freqs = entry.get("quantum_frequencies", [0])
                    freq = np.mean(freqs) if isinstance(freqs, list) and freqs else 0

                    initial_fid = metrics.get("initial_fidelity", 0)
                    final_fid = metrics.get("final_fidelity", 0)
                    improvement = (
                        (final_fid - initial_fid) / initial_fid * 100
                        if initial_fid > 0
                        else 0
                    )

                    print(
                        f"Processing QEC entry - Freq: {freq}, Initial: {initial_fid}, Final: {final_fid}"
                    )  # Debug print

                    frequencies.append(freq)
                    fidelities.append(final_fid)
                    improvements.append(improvement)

            if frequencies and fidelities:
                print(
                    f"Plotting QEC data - Frequencies: {frequencies}, Fidelities: {fidelities}"
                )  # Debug print

                # Plot fidelity
                ax.scatter(
                    frequencies, fidelities, label="Final Fidelity", alpha=0.6, c="blue"
                )
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Fidelity", color="blue")

                # Add second y-axis for improvement percentage
                ax2 = ax.twinx()
                ax2.scatter(
                    frequencies, improvements, label="Improvement %", alpha=0.6, c="red"
                )
                ax2.set_ylabel("Improvement %", color="red")

                ax.grid(True)
                ax.set_title("QEC Performance vs Frequency")

                # Combine legends
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
            else:
                ax.text(
                    0.5, 0.5, "Insufficient QEC data points", ha="center", va="center"
                )

        except Exception as e:
            print(f"Error plotting QEC frequency impact: {str(e)}")
            traceback.print_exc()
            ax.text(
                0.5, 0.5, f"Error plotting data: {str(e)}", ha="center", va="center"
            )

    def _plot_note_qec_correlation(self, ax, data_series):
        """Plot correlation between musical notes and QEC performance"""
        try:
            melody_data = data_series.get("melody", [])
            qec_data = data_series.get("qec", [])

            print(f"Found melody data: {len(melody_data)} entries")  # Debug print
            print(f"Found QEC data: {len(qec_data)} entries")  # Debug print

            if not melody_data or not qec_data:
                ax.text(
                    0.5, 0.5, "No correlation data available", ha="center", va="center"
                )
                return

            notes = []
            qec_effectiveness = []
            fidelities = []

            # Extract notes from melody data
            for entry in melody_data:
                if isinstance(entry, dict):
                    note_data = entry.get("notes", [])
                    if note_data:
                        notes.extend(note_data)
                        print(f"Added notes: {note_data}")  # Debug print

            # Extract QEC metrics
            for entry in qec_data:
                if isinstance(entry, dict):
                    metrics = entry.get("metrics", {})
                    if metrics:
                        initial_fid = metrics.get("initial_fidelity", 0)
                        final_fid = metrics.get("final_fidelity", 0)
                        improvement = (
                            (final_fid - initial_fid) / initial_fid * 100
                            if initial_fid > 0
                            else 0
                        )
                        qec_effectiveness.append(improvement)
                        fidelities.append(final_fid)
                        print(
                            f"Added QEC metrics - Improvement: {improvement}, Fidelity: {final_fid}"
                        )  # Debug print

            if notes and qec_effectiveness:
                # Ensure equal lengths
                min_len = min(len(notes), len(qec_effectiveness))
                print(f"Plotting {min_len} data points")  # Debug print

                x = np.arange(min_len)
                width = 0.35

                # Plot both effectiveness and fidelity
                rects1 = ax.bar(
                    x - width / 2,
                    qec_effectiveness[:min_len],
                    width,
                    label="QEC Improvement %",
                    color="blue",
                    alpha=0.6,
                )
                rects2 = ax.bar(
                    x + width / 2,
                    [f * 100 for f in fidelities[:min_len]],
                    width,
                    label="Final Fidelity %",
                    color="red",
                    alpha=0.6,
                )

                ax.set_xticks(x)
                ax.set_xticklabels(notes[:min_len], rotation=45)
                ax.set_ylabel("Percentage")
                ax.set_title("QEC Performance per Musical Note")
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(
                    0.5, 0.5, "Insufficient correlation data", ha="center", va="center"
                )

        except Exception as e:
            print(f"Error plotting note-QEC correlation: {str(e)}")
            traceback.print_exc()
            ax.text(0.5, 0.5, "Error plotting correlation", ha="center", va="center")

    def collect_and_analyze_data(self):
        """Collect and analyze data from all sources"""
        try:
            # Get results from data manager
            all_results = self.data_manager.get_all_results()

            # Process data into series
            processed_data = {"quantum": [], "qec": [], "melody": [], "fluid": []}

            # Extract frequencies for correlation calculation
            frequencies = {"quantum": [], "melody": [], "fluid": []}

            # Process each system's data
            for system, results in all_results.items():
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict) and "data" in result:
                            processed_data[system].append(result["data"])

                            # Collect frequencies for correlation
                            if (
                                system == "quantum"
                                and "quantum_frequencies" in result["data"]
                            ):
                                frequencies["quantum"].extend(
                                    result["data"]["quantum_frequencies"]
                                )
                            elif (
                                system == "melody"
                                and "quantum_frequencies" in result["data"]
                            ):
                                frequencies["melody"].extend(
                                    result["data"]["quantum_frequencies"]
                                )
                            elif (
                                system == "fluid"
                                and "original_frequencies" in result["data"]
                            ):
                                frequencies["fluid"].extend(
                                    result["data"]["original_frequencies"]
                                )

            # Calculate correlations
            correlations = {}
            for sys1, sys2 in [
                ("quantum", "melody"),
                ("quantum", "fluid"),
                ("melody", "fluid"),
            ]:
                if frequencies[sys1] and frequencies[sys2]:
                    # Ensure equal lengths for correlation
                    min_len = min(len(frequencies[sys1]), len(frequencies[sys2]))
                    corr = np.corrcoef(
                        frequencies[sys1][:min_len], frequencies[sys2][:min_len]
                    )[0, 1]
                    correlations[f"{sys1}_{sys2}"] = corr
                else:
                    correlations[f"{sys1}_{sys2}"] = 0.0

            # Get QEC details including materials
            qec_details = None
            if "qec" in all_results and all_results["qec"]:
                latest_qec = all_results["qec"][-1]["data"]
                qec_details = {
                    "qec_type": latest_qec.get("qec_type", "Unknown"),
                    "material": latest_qec.get("material", {}),
                    "metrics": latest_qec.get("metrics", {}),
                }

            # Run ML analysis
            pattern_analysis = self.data_manager.analyze_ml_patterns()
            analyzed_datapoints = self.data_manager.get_all_results()

            return {
                "data_series": processed_data,
                "pattern_analysis": pattern_analysis,
                "correlations": correlations,
                "qec_details": qec_details,
            }

        except Exception as e:
            print(f"Error in data collection: {str(e)}")
            traceback.print_exc()
            return None

    def _update_analysis_text(self, results, pattern_analysis, analyzed_datapoints):
        try:
            if not hasattr(self, "analysis_text") or not self.analysis_text:
                print("Analysis text widget not available")
                return
            analysis_text = ["=== Comprehensive Analysis Summary ===\n"]

            # Pattern Analysis
            if pattern_analysis:
                analysis_text.extend(
                    [
                        "Pattern Analysis:",
                        f"• Clusters: {len(pattern_analysis.get('clusters', []))}",
                        f"• Features: {pattern_analysis.get('n_features', 0)}",
                        f"• Key Features: {', '.join(list(pattern_analysis.get('feature_importance', {}).keys())[:3])}",
                        f"• Explained Variance: {sum(pattern_analysis.get('explained_variance', [0])) * 100:.2f}%\n",
                    ]
                )

            # Quantum Data
            quantum_data = results.get("data_series", {}).get("quantum", [])
            if quantum_data:
                latest_quantum = quantum_data[-1] if quantum_data else {}
                analysis_text.extend(
                    [
                        "Quantum System State:",
                        f"• State Vector Size: {len(latest_quantum.get('statevector', []))}",
                        f"• State Purity: {latest_quantum.get('purity', 0):.3f}",
                        f"• State Fidelity: {latest_quantum.get('fidelity', 0):.3f}",
                        f"• Number of Frequencies: {len(latest_quantum.get('quantum_frequencies', []))}\n",
                    ]
                )

            # QEC Performance with Materials
            qec_data = results.get("data_series", {}).get("qec", [])
            if qec_data:
                analysis_text.append("QEC Performance:")
                for qec_entry in qec_data:
                    if isinstance(qec_entry, dict):
                        metrics = qec_entry.get("metrics", {})
                        qec_type = qec_entry.get("qec_type", "Unknown")
                        material_info = qec_entry.get("material", {})
                        # material_name = material_info.get("name", "Not specified")
                        temperature = material_info.get("temperature", 0)
                        analysis_text.extend(
                            [
                                f"\nQEC Type: {qec_type}",
                                # f"• Material: {material_name}",
                                f"• Temperature: {temperature}K",
                                f"• Initial Fidelity: {metrics.get('initial_fidelity', 0):.3f}",
                                f"• Final Fidelity: {metrics.get('final_fidelity', 0):.3f}",
                                f"• Recovery Rate: {metrics.get('recovery_rate', 0):.3f}",
                                f"• Error Suppression: {metrics.get('error_suppression', 0):.3f}",
                            ]
                        )

                # if material:
                #     analysis_text.extend(
                #         [
                #             # f"• Material: {material.get('name', 'Unknown')}",
                #             f"• Temperature: {material.get('temperature', 0)}K\n",
                #         ]
                #     )
                else:
                    analysis_text.append("")

            # Melody Analysis
            melody_data = results.get("data_series", {}).get("melody", [])
            if melody_data:
                latest_melody = melody_data[-1] if melody_data else {}
                analysis_text.extend(
                    [
                        "Musical Analysis:",
                        f"• Notes Analyzed: {len(latest_melody.get('notes', []))}",
                        f"• Musical Systems: {', '.join(latest_melody.get('musical_systems', {}).keys())}",
                        f"• Total Frequencies: {len(latest_melody.get('quantum_frequencies', []))}\n",
                    ]
                )

            # Fluid Analysis
            fluid_data = results.get("data_series", {}).get("fluid", [])
            if fluid_data:
                latest_fluid = fluid_data[-1] if fluid_data else {}
                analysis_text.extend(
                    [
                        "Fluid Dynamics:",
                        f"• Original Frequencies: {len(latest_fluid.get('original_frequencies', []))}",
                        f"• Fibonacci Frequencies: {len(latest_fluid.get('fibonacci_frequencies', []))}\n",
                    ]
                )

            # System Correlations
            correlations = results.get("correlations", {})
            analysis_text.extend(
                [
                    "System Correlations:",
                    f"• Quantum-Melody: {correlations.get('quantum_melody', 0):.3f}",
                    f"• Quantum-Fluid: {correlations.get('quantum_fluid', 0):.3f}",
                    f"• Melody-Fluid: {correlations.get('melody_fluid', 0):.3f}\n",
                ]
            )

            # Final safety check before setting text
            if hasattr(self, "analysis_text") and self.analysis_text:
                try:
                    self.analysis_text.setText("\n".join(analysis_text))
                except RuntimeError:
                    print("Widget deleted during text update")
            else:
                print("Widget not available for final update")

        except Exception as e:
            print(f"Error updating analysis text: {str(e)}")

    def calculate_correlations(self, results):
        """Calculate correlations between different systems"""
        try:
            data_series = results.get("data_series", {})
            correlations = {}

            # Extract frequency data from each system
            quantum_freq = []
            melody_freq = []
            fluid_freq = []

            # Get quantum frequencies
            quantum_data = data_series.get("quantum", [])
            if quantum_data:
                latest_quantum = quantum_data[-1]
                if isinstance(latest_quantum, dict):
                    quantum_freq = latest_quantum.get("quantum_frequencies", [])

            # Get melody frequencies
            melody_data = data_series.get("melody", [])
            if melody_data:
                latest_melody = melody_data[-1]
                if isinstance(latest_melody, dict):
                    melody_freq = latest_melody.get("quantum_frequencies", [])

            # Get fluid frequencies
            fluid_data = data_series.get("fluid", [])
            if fluid_data:
                latest_fluid = fluid_data[-1]
                if isinstance(latest_fluid, dict):
                    fluid_freq = latest_fluid.get("original_frequencies", [])

            # Calculate correlations if data exists
            if quantum_freq and melody_freq:
                # Pad shorter array with zeros to match lengths
                max_len = max(len(quantum_freq), len(melody_freq))
                q_freq = np.pad(quantum_freq, (0, max_len - len(quantum_freq)))
                m_freq = np.pad(melody_freq, (0, max_len - len(melody_freq)))
                correlations["quantum_melody"] = np.corrcoef(q_freq, m_freq)[0, 1]

            if quantum_freq and fluid_freq:
                max_len = max(len(quantum_freq), len(fluid_freq))
                q_freq = np.pad(quantum_freq, (0, max_len - len(quantum_freq)))
                f_freq = np.pad(fluid_freq, (0, max_len - len(fluid_freq)))
                correlations["quantum_fluid"] = np.corrcoef(q_freq, f_freq)[0, 1]

            if melody_freq and fluid_freq:
                max_len = max(len(melody_freq), len(fluid_freq))
                m_freq = np.pad(melody_freq, (0, max_len - len(melody_freq)))
                f_freq = np.pad(fluid_freq, (0, max_len - len(fluid_freq)))
                correlations["melody_fluid"] = np.corrcoef(m_freq, f_freq)[0, 1]

            # Handle NaN values
            for key in correlations:
                if np.isnan(correlations[key]):
                    correlations[key] = 0.0

            return correlations

        except Exception as e:
            print(f"Error calculating correlations: {str(e)}")
            return {"quantum_melody": 0.0, "quantum_fluid": 0.0, "melody_fluid": 0.0}
