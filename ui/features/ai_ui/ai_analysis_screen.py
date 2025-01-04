import traceback
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QGroupBox,
    QComboBox,
    QLineEdit,
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.calculations.ai.ai_quantum_interpreter import QuantumInterpreter
from core.calculations.ai.quantum_results_interpreter import QuantumResultsInterpreter
from storage.data_manager import QuantumDataManager


class AIAnalysisScreen(QWidget):
    def __init__(self, data_manager=None):
        super().__init__()
        self.analysis_complete = None
        self.user_input = None
        self.data_manager = data_manager or QuantumDataManager()
        self.interpreter = QuantumResultsInterpreter()
        self.quantum_interpreter = QuantumInterpreter()

        # Initialize UI components
        self.figure = Figure(figsize=(15, 12))
        self.canvas = FigureCanvas(self.figure)
        self.analysis_text = None
        self.data_source_combo = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Header
        header = QLabel("Quantum Systems AI Analysis Dashboard")
        header.setStyleSheet(
            """
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                margin: 10px;
            }
        """
        )
        layout.addWidget(header, alignment=Qt.AlignmentFlag.AlignCenter)

        # Data Source Selection
        data_source_layout = QHBoxLayout()
        data_source_label = QLabel("Select Data Source:")
        data_source_label.setStyleSheet("color: white;")
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(
            [
                "All Results",
                "Quantum Analysis",
                "Melody Analysis",
                "Fluid Dynamics",
                "QEC Analysis",
                "Particle Simulation",  # Add this line
            ]
        )
        data_source_layout.addWidget(data_source_label)
        data_source_layout.addWidget(self.data_source_combo)
        layout.addLayout(data_source_layout)

        # Controls
        controls = QGroupBox()
        controls.setStyleSheet(
            """
            QGroupBox {
                border: none;
                margin-top: 10px;
            }
        """
        )
        controls_layout = QHBoxLayout()

        analyze_btn = QPushButton("Run AI Analysis")
        analyze_btn.clicked.connect(self.analyze_results)
        analyze_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #3d405e;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """
        )
        controls_layout.addWidget(analyze_btn)
        controls.setLayout(controls_layout)
        layout.addWidget(controls)

        # Analysis Results
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                padding: 15px;
                font-size: 14px;
                min-height: 200px;
                margin: 10px;
            }
        """
        )
        layout.addWidget(self.analysis_text)
        # User Input
        user_input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Ask a question about the quantum data...")
        self.user_input.setStyleSheet(
            """
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                padding: 5px;
                font-size: 14px;
            }
        """
        )
        ask_button = QPushButton("Ask AIQuantum")
        ask_button.clicked.connect(self.ask_gpt)
        ask_button.setStyleSheet(
            """
            QPushButton {
                background-color: #3d405e;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """
        )
        user_input_layout.addWidget(self.user_input)
        user_input_layout.addWidget(ask_button)
        layout.addLayout(user_input_layout)

        # Visualization
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def analyze_results(self):
        try:
            all_results = self.data_manager.get_all_results()
            pattern_analysis = self.data_manager.analyze_data_patterns()

            analysis_text = ["=== Comprehensive Analysis ===\n"]

            # Add ML insights
            if pattern_analysis:
                analysis_text.append("Machine Learning Patterns:")
                analysis_text.append(
                    f"• Clusters Identified: {len(set(pattern_analysis['clusters']))}"
                )
                analysis_text.append(
                    f"• Anomalies Found: {pattern_analysis['anomalies'].count(-1)}"
                )
                analysis_text.append("• Cluster Analysis:")
                for i, cluster in enumerate(set(pattern_analysis["clusters"])):
                    cluster_data = [
                        j
                        for j, c in enumerate(pattern_analysis["clusters"])
                        if c == cluster
                    ]
                    analysis_text.append(
                        f"  - Cluster {i + 1}: {len(cluster_data)} points"
                    )

            # Get interpreted results
            summary = self.interpreter.generate_summary(
                {
                    **all_results,
                    "ml_analysis": pattern_analysis if pattern_analysis else {},
                }
            )

            analysis_text.append("\nDetailed Analysis:")
            for result_type, results in all_results.items():
                if results:
                    latest = results[-1]["data"]
                    analysis_text.append(f"\n{result_type.capitalize()} Analysis:")

                    if result_type == "quantum":
                        analysis_text.append(f"• Purity: {latest.get('purity', 'N/A')}")
                        analysis_text.append(
                            f"• Fidelity: {latest.get('fidelity', 'N/A')}"
                        )
                    elif result_type == "qec":
                        metrics = latest.get("metrics", {})
                        analysis_text.append(
                            f"• Initial Fidelity: {metrics.get('initial_fidelity', 'N/A')}"
                        )
                        analysis_text.append(
                            f"• Final Fidelity: {metrics.get('final_fidelity', 'N/A')}"
                        )
                    elif result_type == "melody":
                        systems = latest.get("musical_systems", {})
                        for system, data in systems.items():
                            analysis_text.append(
                                f"• {system}: {len(data.get('notes', []))} notes"
                            )

            # Add summary
            analysis_text.append(f"\nAI Interpretation:\n{summary}")

            self.analysis_text.setText("\n".join(analysis_text))

            # Update visualizations
            if pattern_analysis:
                self.update_visualizations(pattern_analysis, all_results)

        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            traceback.print_exc()

    # def analyze_results(self):
    #     try:
    #         print("\nStarting AI Analysis...")
    #         all_results = self.data_manager.get_all_results()
    #
    #         if not any(all_results.values()):
    #             self.analysis_text.setText(
    #                 "No data available for analysis. Please run some analyses first."
    #             )
    #             return
    #
    #         # Get a summary from the interpreter
    #         summary = self.interpreter.generate_summary(all_results)
    #
    #         # Get detailed interpretation
    #         interpretation = self.quantum_interpreter.interpret_quantum_data(
    #             all_results
    #         )
    #
    #         # Get latest results for display
    #         latest_results = all_results
    #         for data_type, result_list in latest_results.items():
    #             if result_list and len(result_list) > 0:
    #                 latest_results[data_type] = result_list[-1]["data"]
    #
    #         # Display both regular analysis and AI interpretation
    #         analysis_text = [
    #             "\n=== AI Interpretation ===",
    #             f"\nSummary: {summary}",
    #             f"\nDetailed Analysis: {interpretation}",
    #         ]
    #
    #         self.analysis_text.setText("\n".join(analysis_text))
    #
    #         # Update visualizations
    #         self.update_visualizations(latest_results)
    #
    #         # Update results to AI
    #         self.display_and_analyze_results(latest_results)
    #
    #     except Exception as e:
    #         error_message = f"Error performing analysis: {str(e)}"
    #         print(error_message)
    #         traceback.print_exc()
    #         self.analysis_text.setText(error_message)

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

    def _plot_harmonic_patterns(self, ax, results):
        frequencies = []

        # Extract frequencies from different sources
        if "quantum" in results and "frequencies" in results["quantum"]:
            frequencies.extend(results["quantum"]["frequencies"])
        if "melody" in results and "frequencies" in results["melody"]:
            frequencies.extend(results["melody"]["frequencies"])
        if "fluid" in results and "original_frequencies" in results["fluid"]:
            frequencies.extend(results["fluid"]["original_frequencies"])

        if frequencies:
            ax.stem(range(len(frequencies)), frequencies)
            ax.set_title("Harmonic Frequency Patterns")
            ax.set_xlabel("Frequency Index")
            ax.set_ylabel("Frequency (Hz)")
        else:
            ax.text(0.5, 0.5, "No Harmonic Data", ha="center", va="center")

    def _plot_state_evolution(self, ax, results):
        # Plot state evolution from various sources
        if "quantum" in results and "statevector" in results["quantum"]:
            statevector = np.array(results["quantum"]["statevector"])
            ax.plot(np.abs(statevector), label="Quantum State")
            ax.set_title("State Evolution")
            ax.set_xlabel("State Index")
            ax.set_ylabel("Magnitude")
            ax.legend()
        elif "particle" in results and "positions" in results["particle"]:
            positions = np.array(results["particle"]["positions"])
            ax.plot(np.linalg.norm(positions, axis=1), label="Particle Positions")
            ax.set_title("Particle Evolution")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Position Magnitude")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No Evolution Data", ha="center", va="center")

    def _plot_quantum_state(self, ax, data):
        """Plot quantum state if available"""
        if "statevector" in data and data["statevector"] is not None:
            statevector = np.array(data["statevector"])
            probabilities = np.abs(statevector) ** 2
            ax.bar(range(len(probabilities)), probabilities)
            ax.set_title("Quantum State Probabilities")
            ax.set_xlabel("State")
            ax.set_ylabel("Probability")

    def _plot_correlation_matrix(self, ax, results):
        all_data = []
        labels = []

        # Collect data from different sources
        if "quantum" in results and "frequencies" in results["quantum"]:
            all_data.append(results["quantum"]["frequencies"])
            labels.append("Quantum")
        if "melody" in results and "frequencies" in results["melody"]:
            all_data.append(results["melody"]["frequencies"])
            labels.append("Melody")
        if "fluid" in results and "original_frequencies" in results["fluid"]:
            all_data.append(results["fluid"]["original_frequencies"])
            labels.append("Fluid")

        if len(all_data) > 1:
            # Pad arrays to same length
            max_len = max(len(d) for d in all_data)
            padded_data = [
                np.pad(d, (0, max_len - len(d)), "constant") for d in all_data
            ]

            corr_matrix = np.corrcoef(padded_data)
            im = ax.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
            ax.set_title("Cross-System Correlations")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            plt.colorbar(im, ax=ax)
        else:
            ax.text(
                0.5, 0.5, "Insufficient Data for Correlation", ha="center", va="center"
            )

    def _plot_quality_metrics(self, ax, results):
        metrics = []
        labels = []

        # Collect quality metrics from different sources
        if "quantum" in results:
            if "purity" in results["quantum"]:
                metrics.append(results["quantum"]["purity"])
                labels.append("Quantum Purity")
            if "fidelity" in results["quantum"]:
                metrics.append(results["quantum"]["fidelity"])
                labels.append("Quantum Fidelity")

        if "qec" in results and "metrics" in results["qec"]:
            qec_metrics = results["qec"]["metrics"]
            if "initial_fidelity" in qec_metrics:
                metrics.append(qec_metrics["initial_fidelity"])
                labels.append("Initial QEC Fidelity")
            if "final_fidelity" in qec_metrics:
                metrics.append(qec_metrics["final_fidelity"])
                labels.append("Final QEC Fidelity")

        if metrics:
            ax.bar(range(len(metrics)), metrics)
            ax.set_title("System Quality Metrics")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Metric Value")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)
        else:
            ax.text(0.5, 0.5, "No Quality Metrics Available", ha="center", va="center")

    def ask_gpt(self):
        try:
            question = self.user_input.text()
            if not question:
                question = "What does this quantum system tell us?"

            # Get detailed data
            latest_results = self.data_manager.get_latest_results()
            pattern_analysis = self.data_manager.analyze_data_patterns()

            data_summary = ["=== Detailed Quantum System Analysis ==="]

            # Pattern Analysis Details
            if pattern_analysis and isinstance(pattern_analysis, dict):
                data_summary.extend(
                    [
                        "\nMachine Learning Pattern Details:",
                        "Cluster Analysis:",
                        f"- Found {len(set(pattern_analysis.get('clusters', [])))} distinct behavioral patterns",
                    ]
                )

                # Add cluster descriptions
                clusters = pattern_analysis.get("clusters", [])
                for i, cluster_id in enumerate(set(clusters)):
                    count = clusters.count(cluster_id)
                    data_summary.append(
                        f"- Cluster {i + 1} ({count} points): "
                        + "Represents a distinct quantum state pattern"
                    )

                # Feature importance
                if "feature_importance" in pattern_analysis:
                    data_summary.append("\nKey System Features:")
                    for feature, importance in pattern_analysis[
                        "feature_importance"
                    ].items():
                        data_summary.append(
                            f"- {feature}: Impact Score {importance:.3f}"
                        )

            # System-specific Details
            if latest_results:
                for system, result in latest_results.items():
                    if result and isinstance(result, dict) and "data" in result:
                        data = result["data"]

                        if system == "qec":
                            metrics = data.get("metrics", {})
                            init_fid = metrics.get("initial_fidelity", 0)
                            final_fid = metrics.get("final_fidelity", 0)
                            data_summary.extend(
                                [
                                    "\nQuantum Error Correction Details:",
                                    f"- Starting System Fidelity: {init_fid:.3f}",
                                    f"- Final System Fidelity: {final_fid:.3f}",
                                    f"- Fidelity Change: {final_fid - init_fid:.3f}",
                                    f"- Error Correction Performance: {abs((final_fid / init_fid) * 100):.1f}% maintained",
                                ]
                            )

                        elif system == "melody":
                            if "musical_systems" in data:
                                data_summary.append("\nQuantum-Musical Correlations:")
                                for sys_name, sys_data in data[
                                    "musical_systems"
                                ].items():
                                    notes = sys_data.get("notes", [])
                                    data_summary.extend(
                                        [
                                            f"\n{sys_name} System:",
                                            f"- Notes Present: {', '.join(notes)}",
                                            f"- Total Notes: {len(notes)}",
                                            f"- Frequencies Used: {sys_data.get('frequencies', [])}",
                                        ]
                                    )

                        elif system == "fluid":
                            orig_freq = data.get("original_frequencies", [])
                            fib_freq = data.get("fibonacci_frequencies", [])
                            data_summary.extend(
                                [
                                    "\nFluid Dynamics Analysis:",
                                    f"- Base Frequencies: {orig_freq}",
                                    f"- Harmonic Expansions: {len(fib_freq)} Fibonacci frequencies",
                                    f"- Frequency Range: {min(orig_freq) if orig_freq else 'N/A'} to {max(orig_freq) if orig_freq else 'N/A'} Hz",
                                ]
                            )

                        elif system == "particle":
                            pos = data.get("positions", [])
                            vel = data.get("velocities", [])
                            data_summary.extend(
                                [
                                    "\nParticle Behavior:",
                                    f"- Analysis Mode: {data.get('mode', 'N/A')}",
                                    f"- Particle Count: {len(pos)}",
                                    f"- Average Velocity: {np.mean(vel) if vel else 'N/A'}",
                                ]
                            )

            # Create detailed prompt
            prompt = (
                f"Based on this detailed quantum system analysis:\n"
                f"{chr(10).join(data_summary)}\n\n"
                f"Question: {question}\n"
                "Please provide a comprehensive analysis explaining what these patterns "
                "and measurements indicate about the quantum system's behavior, stability, "
                "and correlations between different aspects (musical, fluid, particle)."
            )

            response = self.quantum_interpreter.interpret_quantum_data(
                {"prompt": prompt}
            )

            self.analysis_text.append(f"\nQ: {question}")
            self.analysis_text.append(f"A: {response}")

        except Exception as e:
            print(f"Error in AI analysis: {str(e)}")
            traceback.print_exc()
            self.analysis_text.append(f"\nError processing query: {str(e)}")

    def update_visualizations(self, pattern_analysis, all_results):
        self.figure.clear()
        # Increase spacing between subplots
        gs = self.figure.add_gridspec(2, 2, hspace=0.5, wspace=0.4)

        # Plot ML analysis results (top row)
        if pattern_analysis and "pca_components" in pattern_analysis:
            # Clusters plot
            ax1 = self.figure.add_subplot(gs[0, 0])
            X_pca = np.array(pattern_analysis["pca_components"])
            clusters = np.array(pattern_analysis["clusters"])
            scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis")
            ax1.set_title("Quantum State Clusters")
            ax1.legend(
                *scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1)
            )

            # Anomalies plot
            ax2 = self.figure.add_subplot(gs[0, 1])
            anomalies = np.array(pattern_analysis["anomalies"])
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=anomalies, cmap="RdYlBu")
            ax2.set_title("Anomaly Detection")

        # Get latest results
        latest_results = {}
        for k, v in all_results.items():
            if v and len(v) > 0:
                latest_results[k] = v[-1]["data"]

        # Correlation Matrix with adjusted position
        ax3 = self.figure.add_subplot(gs[1, 0])
        self._plot_correlation_matrix(ax3, latest_results)
        ax3.set_title("System Correlations", pad=20)

        # Quality Metrics with adjusted position
        ax4 = self.figure.add_subplot(gs[1, 1])
        self._plot_quality_metrics(ax4, latest_results)
        ax4.set_title("Quality Metrics", pad=20)

        # Adjust layout to prevent overlap
        self.figure.tight_layout()
        self.canvas.draw()

    # def update_visualizations(self, pattern_analysis, all_results):
    #     self.figure.clear()
    #     gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    #     results = self.data_manager.get_all_results()
    #
    #     if pattern_analysis and "pca_components" in pattern_analysis:
    #         # Clusters plot
    #         ax1 = self.figure.add_subplot(gs[0, 0])
    #         X_pca = np.array(pattern_analysis["pca_components"])
    #         clusters = np.array(pattern_analysis["clusters"])
    #         scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis")
    #         ax1.set_title("Quantum State Clusters")
    #         ax1.legend(*scatter.legend_elements(), title="Clusters")
    #
    #         # Anomalies plot
    #         ax2 = self.figure.add_subplot(gs[0, 1])
    #         anomalies = np.array(pattern_analysis["anomalies"])
    #         ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=anomalies, cmap="RdYlBu")
    #         ax2.set_title("Anomaly Detection")
    #
    #     # Plot PCA results
    #     ax1 = self.figure.add_subplot(gs[0, 0])
    #     X_pca = pattern_analysis["pca_components"]
    #     clusters = pattern_analysis["clusters"]
    #     ax1.scatter([x[0] for x in X_pca], [x[1] for x in X_pca], c=clusters)
    #     ax1.set_title("Quantum State Clusters")
    #
    #     if not any(results.values()):
    #         self.analysis_text.setText(
    #             "No data available for visualizations. Please run some analyses first."
    #         )
    #         return
    #
    #     # Get latest results for each type
    #     latest_results = results
    #     for data_type, result_list in results.items():
    #         if result_list:  # Check if the list is not empty
    #             latest_results[data_type] = result_list[-1]["data"]
    #
    #     # Harmonic Patterns
    #     ax1 = self.figure.add_subplot(gs[0, 0])
    #     self._plot_harmonic_patterns(ax1, latest_results)
    #
    #     # State Evolution
    #     ax2 = self.figure.add_subplot(gs[0, 1])
    #     self._plot_state_evolution(ax2, latest_results)
    #
    #     # Correlation Matrix
    #     ax3 = self.figure.add_subplot(gs[1, 0])
    #     self._plot_correlation_matrix(ax3, latest_results)
    #
    #     # Quality Metrics
    #     ax4 = self.figure.add_subplot(gs[1, 1])
    #     self._plot_quality_metrics(ax4, latest_results)
    #
    #     # Plot data analysis results (bottom row)
    #     if all_results:
    #         # Get latest results
    #         latest_results = {
    #             k: results[-1]["data"] if results else None
    #             for k, results in all_results.items()
    #         }
    #
    #         # Correlation Matrix
    #         ax3 = self.figure.add_subplot(gs[1, 0])
    #         self._plot_correlation_matrix(ax3, latest_results)
    #
    #         # Quality Metrics
    #         ax4 = self.figure.add_subplot(gs[1, 1])
    #         self._plot_quality_metrics(ax4, latest_results)
    #
    #     self.canvas.draw()
