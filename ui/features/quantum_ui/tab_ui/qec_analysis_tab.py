import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSlider,
    QPushButton,
    QTextEdit,
    QDialog,
    QScrollArea,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from core.calculations.qec.qec_analysis import QECAnalyzer
from core.materials.materials_database import (
    get_material_properties,
    get_materials_by_category,
    get_all_materials,
    get_material_categories,
)
from storage.data_manager import QuantumDataManager
import qtawesome as qta


class QECAnalysisTab(QWidget):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_manager = QuantumDataManager()
        self.more_info_btn = None
        self.results_text = None
        self.canvas = None
        self.figure = None
        self.qec_selector = None
        self.analyze_btn = None
        self.t1_label = None
        self.t2_label = None
        self.t2_slider = None
        self.t1_slider = None
        self.analyzer = QECAnalyzer()
        self.import_fluid_btn = QPushButton("Import Fluid Data")
        self.import_fluid_btn.clicked.connect(self.import_fluid_data)

        # New material-related attributes
        self.category_selector = None
        self.material_selector = None
        self.properties_text = None
        self.temperature_slider = None
        self.temp_label = None

        # Additional analysis attributes
        self.last_analysis_results = None
        self.current_material_properties = None

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # QEC Controls group
        qec_group = QGroupBox("QEC Controls")
        qec_group.setStyleSheet(
            """
            QGroupBox {
                color: white;
                font-size: 14px;
                border: 1px solid #3d405e;
                border-radius: 5px;
                padding: 15px;
                background-color: rgba(61, 64, 94, 0.3);
            }
            QLabel { color: white; }
        """
        )
        qec_layout = QVBoxLayout()
        qec_layout.setSpacing(5)

        self.more_info_btn = QPushButton()
        self.more_info_btn.setIcon(
            qta.icon("fa.question-circle", color="#2196F3")
        )  # Set blue color
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
        qec_layout.addWidget(self.more_info_btn)

        # Circuit type selector
        type_layout = QHBoxLayout()
        type_label = QLabel("Error Correction Code:")
        self.qec_selector = QComboBox()
        self.qec_selector.addItems(
            [
                "3-Qubit Bit Flip",
                "3-Qubit Phase Flip",
                "5-Qubit Code",
                "7-Qubit Steane",
                "9-Qubit Shor",
            ]
        )
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.qec_selector)
        qec_layout.addLayout(type_layout)

        # T1/T2 Controls
        params_group = QGroupBox("Decoherence Parameters")
        params_group.setStyleSheet("color: white; border: 1px solid #3d405e;")
        params_layout = QVBoxLayout()

        # T1 control
        t1_layout = QHBoxLayout()
        t1_label = QLabel("T1:")
        self.t1_slider = QSlider(Qt.Orientation.Horizontal)
        self.t1_slider.setRange(1, 5000)  # Now 1-5000 ms instead of 1-100 μs
        self.t1_slider.setValue(500)  # Default to 500 ms
        self.t1_label = QLabel("500 ms")  # Updated to show ms
        t1_layout.addWidget(t1_label)
        t1_layout.addWidget(self.t1_slider)
        t1_layout.addWidget(self.t1_label)
        self.t1_slider.valueChanged.connect(lambda v: self.t1_label.setText(f"{v} ms"))
        params_layout.addLayout(t1_layout)

        # T2 control
        t2_layout = QHBoxLayout()
        t2_label = QLabel("T2:")
        self.t2_slider = QSlider(Qt.Orientation.Horizontal)
        self.t2_slider.setRange(1, 2500)  # Now 1-2500 ms
        self.t2_slider.setValue(250)  # Default to 250 ms
        self.t2_label = QLabel("250 ms")  # Updated to show ms
        t2_layout.addWidget(t2_label)
        t2_layout.addWidget(self.t2_slider)
        t2_layout.addWidget(self.t2_label)
        self.t2_slider.valueChanged.connect(lambda v: self.t2_label.setText(f"{v} ms"))
        params_layout.addLayout(t2_layout)

        params_group.setLayout(params_layout)
        qec_layout.addWidget(params_group)
        qec_group.setLayout(qec_layout)

        # Material Controls group
        material_group = QGroupBox("Material Controls")
        material_group.setStyleSheet(
            """
            QGroupBox {
                color: white;
                font-size: 14px;
                border: 1px solid #3d405e;
                border-radius: 5px;
                padding: 15px;
                background-color: rgba(61, 64, 94, 0.3);
            }
            QLabel { color: white; }
        """
        )
        material_layout = QVBoxLayout()

        # Material selector
        material_select_layout = QHBoxLayout()
        material_label = QLabel("Material:")
        self.material_selector = QComboBox()
        self.material_selector.addItems(get_all_materials())
        material_select_layout.addWidget(material_label)
        material_select_layout.addWidget(self.material_selector)
        material_layout.addLayout(material_select_layout)

        # Temperature control
        temp_layout = QHBoxLayout()
        temp_label = QLabel("Temperature (K):")
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(4, 500)
        self.temperature_slider.setValue(300)
        self.temp_label = QLabel("300 K")
        self.temperature_slider.valueChanged.connect(
            lambda v: self.temp_label.setText(f"{v} K")
        )
        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.temperature_slider)
        temp_layout.addWidget(self.temp_label)
        material_layout.addLayout(temp_layout)

        material_group.setLayout(material_layout)

        # Analysis button
        self.analyze_btn = QPushButton("Run QEC Analysis")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #3d405e;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """
        )

        # Add all components to main layout
        layout.addWidget(qec_group)
        layout.addWidget(material_group)
        layout.addWidget(self.analyze_btn)

        # Plot area
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                padding: 8px;
                font-family: monospace;
                font-size: 12px;
                color: #000000;
            }
        """
        )
        layout.addWidget(self.results_text)

        self.setLayout(layout)

    def run_analysis(self):
        try:
            qec_type = self.qec_selector.currentText()
            t1 = self.t1_slider.value()
            t2 = min(self.t2_slider.value(), 2 * t1)

            material = self.material_selector.currentText()
            material_props = None
            if hasattr(self, "material_selector") and material:
                temperature = self.temperature_slider.value()
                material_props = get_material_properties(material)
                material_props["temperature"] = temperature

            self.results_text.clear()
            self.results_text.append(f"Running analysis for {qec_type}...")

            results = self.analyzer.analyze_circuit(qec_type, t1, t2, material_props)

            # Format results for AI analysis
            ai_compatible_results = {
                "frequencies": results["simulation_results"].get("time_points", []),
                "purity": results["metrics"].get("final_fidelity", 0),
                "fidelity": results["metrics"].get("final_fidelity", 0),
                "qec_type": qec_type,
                "t1": t1,
                "t2": t2,
                "material": material_props,
                "metrics": results["metrics"],
                "simulation_results": results["simulation_results"],
            }

            # Store results in data manager
            self.data_manager.update_qec_results(ai_compatible_results)

            # Emit the AI-compatible results
            self.analysis_complete.emit(ai_compatible_results)

            # Plot and display using original results
            self.plot_results(results)
            self.display_results(results)

        except Exception as e:
            self.results_text.append(
                f"\nError during analysis: {str(e)}"
            )  # Fixed extra parenthesis

    def plot_results(self, results):
        """Plot the analysis results with millisecond time scale"""
        self.figure.clear()

        # Create two subplots
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)

        # Plot fidelity over time
        sim_results = results["simulation_results"]
        times = sim_results["time_points"]
        fidelities = [r["fidelity"] for r in sim_results["results"]]

        ax1.plot(times, fidelities, "b-", label="With QEC")
        ax1.plot(
            times,
            [r["theoretical_decay"] for r in sim_results["results"]],
            "r--",
            label="Without QEC",
        )
        ax1.set_xlabel("Time (ms)")  # Updated to milliseconds
        ax1.set_ylabel("Fidelity")
        ax1.set_title("Decoherence Effects")
        ax1.legend()

        # Plot final state distribution
        final_counts = sim_results["results"][-1]["counts"]
        states = list(final_counts.keys())
        counts = list(final_counts.values())
        ax2.bar(range(len(states)), counts)
        ax2.set_xticks(range(len(states)))
        ax2.set_xticklabels(states, rotation=45)
        ax2.set_title("Final State Distribution")

        self.figure.tight_layout()
        self.canvas.draw()

    def display_results(self, results):
        """Display the analysis results with millisecond time scale"""
        metrics = results["metrics"]
        params = results["params"]

        output = [
            "=== QEC Analysis Results ===\n",
            f"QEC Type: {params['qec_type']}",
            f"\nDecoherence Parameters:",
            f"T1 (Relaxation): {params['t1']} ms",  # Updated to milliseconds
            f"T2 (Dephasing): {params['t2']} ms",  # Updated to milliseconds
            f"\nQEC Performance:",
            f"Initial Fidelity: {metrics['initial_fidelity']:.3f}",
            f"Final Fidelity: {metrics['final_fidelity']:.3f}",
            f"Improvement: {metrics['improvement']:+.1f}%",
        ]

        self.results_text.append("\n".join(output))

    def import_fluid_data(self, fluid_data):
        try:
            if not isinstance(fluid_data, dict):
                raise ValueError("Invalid fluid data format")

            frequencies = fluid_data.get("original_frequencies", [])
            if not frequencies:
                self.results_text.append("No frequency data available")
                return

            # Calculate parameters from frequencies
            t1 = max(1, int(np.mean(frequencies) * 1e-6))  # Ensure minimum value of 1
            t2 = max(1, min(t1 // 2, self.t2_slider.value()))

            # Update UI
            self.t1_slider.setValue(t1)
            self.t2_slider.setValue(t2)

            self.results_text.append("Successfully imported fluid data")
            self.results_text.append(f"Number of frequencies: {len(frequencies)}")
            self.results_text.append(f"T1: {t1:.2f} µs")
            self.results_text.append(f"T2: {t2:.2f} µs")

        except Exception as e:
            self.results_text.append(f"Error importing fluid data: {str(e)}")

    def setup_material_controls(self):
        group = QGroupBox("Material Parameters")
        group.setStyleSheet(
            """
            QGroupBox {
                color: white;
                font-size: 14px;
                border: 1px solid #3d405e;
                border-radius: 5px;
                padding: 15px;
                background-color: rgba(61, 64, 94, 0.3);
            }
            QLabel { color: white; }
            QComboBox {
                background-color: white;
                padding: 5px;
                border-radius: 3px;
            }
        """
        )

        layout = QVBoxLayout()

        # Category selector
        category_layout = QHBoxLayout()
        category_label = QLabel("Category:")
        self.category_selector = QComboBox()
        self.category_selector.addItems(["All"] + get_material_categories())
        self.category_selector.currentTextChanged.connect(self.update_material_list)
        category_layout.addWidget(category_label)
        category_layout.addWidget(self.category_selector)
        layout.addLayout(category_layout)

        # Material selector
        material_layout = QHBoxLayout()
        material_label = QLabel("Material:")
        self.material_selector = QComboBox()
        self.material_selector.addItems(get_all_materials())
        self.material_selector.currentTextChanged.connect(self.update_material_info)
        material_layout.addWidget(material_label)
        material_layout.addWidget(self.material_selector)
        layout.addLayout(material_layout)

        # Material properties display
        self.properties_text = QTextEdit()
        self.properties_text.setReadOnly(True)
        self.properties_text.setMaximumHeight(100)
        self.properties_text.setStyleSheet(
            """
            QTextEdit {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
                border: 1px solid #3d405e;
                border-radius: 3px;
            }
        """
        )
        layout.addWidget(self.properties_text)

        # Temperature control
        temp_layout = QHBoxLayout()
        temp_label = QLabel("Temperature (K):")
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(4, 500)
        self.temperature_slider.setValue(300)
        self.temp_label = QLabel("300 K")
        self.temperature_slider.valueChanged.connect(
            lambda v: self.temp_label.setText(f"{v} K")
        )
        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.temperature_slider)
        temp_layout.addWidget(self.temp_label)
        layout.addLayout(temp_layout)

        group.setLayout(layout)
        return group

    def update_material_list(self):
        """Update material list based on selected category"""
        self.material_selector.clear()
        category = self.category_selector.currentText()

        if category == "All":
            materials = get_all_materials()
        else:
            materials = get_materials_by_category(category)

        self.material_selector.addItems(materials)

    def update_material_info(self, material):
        """Update material information display and temperature range"""
        if not material:
            return

        properties = get_material_properties(material)

        # Update properties display
        info_text = f"Properties for {material}:\n"
        for key, value in properties.items():
            if key != "category":  # Skip category in display
                info_text += f"{key}: {value:g}\n"

        self.properties_text.setText(info_text)

        # Update temperature slider range
        max_temp = min(500, properties["max_temp"])
        self.temperature_slider.setRange(4, max_temp)

        # Special handling for superconductors
        if properties["category"] == "superconductor":
            self.temperature_slider.setValue(min(properties["critical_temp"], 300))

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
