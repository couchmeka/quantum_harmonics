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
                "Circuit",
                "Melody Analysis",
                "Fluid Dynamics",
                "QEC Analysis",
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
            print("\nStarting AI Analysis...")
            all_results = self.data_manager.get_all_results()
            selected_source = self.data_source_combo.currentText()

            # Map combobox text to result keys
            source_map = {
                "All Results": None,
                "Quantum Analysis": "quantum",
                "Circuit": "quantum",
                "Melody Analysis": "melody",
                "Fluid Dynamics": "fluid",
                "QEC Analysis": "qec",
            }

            # Get the key for the selected source
            source_key = source_map.get(selected_source)

            # Filter results based on selection
            if source_key is None:
                filtered_results = {
                    k: v for k, v in all_results.items() if v is not None
                }
                if not filtered_results:
                    self.analysis_text.setText(
                        "No analysis results available. Please run analysis in any tab first."
                    )
                    return
                print("Using all available results")
            else:
                result_data = all_results.get(source_key)
                if result_data:
                    filtered_results = {source_key: result_data}
                    print(f"Using {source_key} results")
                else:
                    self.analysis_text.setText(
                        f"No results available for {selected_source}. Please run analysis in that tab first."
                    )
                    return

            print("Generating analysis...")
            # Get AI interpretation
            summary = self.interpreter.generate_summary(filtered_results)
            interpretation = self.quantum_interpreter.interpret_quantum_data(
                filtered_results
            )

            # Display results
            analysis_text = (
                f"Summary:\n{summary}\n\nDetailed Analysis:\n{interpretation}"
            )
            print("Analysis complete")
            self.analysis_text.setText(analysis_text)

            # Update visualizations
            self.update_visualizations(filtered_results)

        except Exception as e:
            error_message = f"Error during analysis: {str(e)}"
            print(error_message)
            self.analysis_text.setText(error_message)
            import traceback

            traceback.print_exc()

    def ask_gpt(self):
        try:
            question = self.user_input.text()
            if not question:
                return

            # Get the current selected data source
            selected_source = self.data_source_combo.currentText()
            filtered_results = self.get_filtered_results(selected_source)

            if not filtered_results:
                self.analysis_text.append(
                    "\nNo data available for analysis. Please run analysis first."
                )
                return

            # Construct a prompt for GPT
            prompt = (
                f"Based on the following quantum data:\n"
                f"{filtered_results}\n\n"
                f"Question: {question}\nAnswer:"
            )

            # Get GPT's response
            response = self.quantum_interpreter.interpret_quantum_data(
                {"prompt": prompt}
            )

            # Display the response
            self.analysis_text.append(f"\nQ: {question}\nA: {response}")

        except Exception as e:
            error_message = f"\nError processing query: {str(e)}"
            print(error_message)
            self.analysis_text.append(error_message)
            import traceback

            traceback.print_exc()

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
            }

            key = source_map.get(selected_source)
            if key and key in all_results and all_results[key]:
                return {key: all_results[key]}
            return {}

        except Exception as e:
            print(f"Error filtering results: {str(e)}")
            return {}

    def _plot_harmonic_patterns(self, ax, results):
        ax.clear()
        frequencies = []

        # Extract frequencies from standardized data structure
        for analysis_type, data in results.items():
            if data and "frequencies" in data:
                frequencies.extend(data["frequencies"])
            elif data and "original_frequencies" in data:
                frequencies.extend(data["original_frequencies"])

        if frequencies:
            ax.stem(range(len(frequencies)), frequencies)
            ax.set_title("Harmonic Frequency Patterns")
            ax.set_xlabel("Frequency Index")
            ax.set_ylabel("Frequency (Hz)")
        else:
            ax.text(
                0.5,
                0.5,
                "No Harmonic Data",
                horizontalalignment="center",
                verticalalignment="center",
            )

    def update_visualizations(self, results):
        self.figure.clear()

        # Create four visualization panels
        gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Harmonic Patterns
        ax1 = self.figure.add_subplot(gs[0, 0])
        self._plot_harmonic_patterns(ax1, results)

        # State Evolution
        ax2 = self.figure.add_subplot(gs[0, 1])
        self._plot_state_evolution(ax2, results)

        # Correlation Matrix
        ax3 = self.figure.add_subplot(gs[1, 0])
        self._plot_correlation_matrix(ax3, results)

        # Quality Metrics
        ax4 = self.figure.add_subplot(gs[1, 1])
        self._plot_quality_metrics(ax4, results)

        self.canvas.draw()

    def _plot_state_evolution(self, ax, results):
        ax.clear()
        # Plot state evolution from various sources
        if results.get("quantum_results", {}).get("statevector"):
            statevector = results["quantum_results"]["statevector"]
            ax.plot(np.abs(statevector), label="Quantum State")
        elif results.get("particle_results", {}).get("positions"):
            positions = results["particle_results"]["positions"]
            ax.plot(np.linalg.norm(positions, axis=1), label="Particle Trajectories")
        else:
            ax.text(
                0.5,
                0.5,
                "No State Evolution Data",
                horizontalalignment="center",
                verticalalignment="center",
            )
        ax.set_title("State Evolution")

    def _plot_correlation_matrix(self, ax, results):
        ax.clear()
        # Attempt to create correlation matrix from various results
        all_data = []
        if results.get("quantum_results", {}).get("frequencies"):
            all_data.append(results["quantum_results"]["frequencies"])
        if results.get("melody_results", {}).get("frequencies"):
            all_data.append(results["melody_results"]["frequencies"])

        if all_data and len(all_data) > 1:
            corr_matrix = np.corrcoef(all_data)
            im = ax.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
            ax.set_title("Data Correlation Matrix")
            plt.colorbar(im, ax=ax)
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient Data for Correlation",
                horizontalalignment="center",
                verticalalignment="center",
            )

    def _plot_quality_metrics(self, ax, results):
        ax.clear()
        metrics = []
        labels = []

        # Collect quality metrics from various sources
        if results.get("quantum_results"):
            metrics.append(results["quantum_results"].get("purity", 0))
            labels.append("Quantum Purity")

        if results.get("qec_results", {}).get("metrics"):
            qec_metrics = results["qec_results"]["metrics"]
            metrics.extend(
                [
                    qec_metrics.get("initial_fidelity", 0),
                    qec_metrics.get("final_fidelity", 0),
                ]
            )
            labels.extend(["Initial Fidelity", "Final Fidelity"])

        if metrics:
            ax.bar(labels, metrics)
            ax.set_title("System Quality Metrics")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Metric Value")
        else:
            ax.text(
                0.5,
                0.5,
                "No Quality Metrics Available",
                horizontalalignment="center",
                verticalalignment="center",
            )
