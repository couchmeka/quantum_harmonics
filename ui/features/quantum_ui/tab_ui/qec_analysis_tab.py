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
    QGridLayout,
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
        layout = QVBoxLayout()  # Main layout for the widget/window

        # Input controls group box
        input_group = QGroupBox("QEC Analysis Controls")
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

        # Grid layout for controls
        input_layout = QGridLayout()

        # Create a More Info button with reference styling
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

        # Material selection
        material_label = QLabel("Material:")
        material_label.setStyleSheet("color: white;")
        self.material_selector = QComboBox()
        self.material_selector.addItems(get_all_materials())

        # QEC selection
        qec_label = QLabel("Error Correction Code:")
        qec_label.setStyleSheet("color: white;")
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

        # T1 control with reference styling
        t1_label = QLabel("T1:")
        t1_label.setStyleSheet("color: white;")
        self.t1_slider = QSlider(Qt.Orientation.Horizontal)
        self.t1_slider.setRange(1, 5000)
        self.t1_slider.setValue(500)
        self.t1_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.t1_slider.setTickInterval(500)
        self.t1_label = QLabel("500 ms")
        self.t1_label.setStyleSheet("color: white;")
        self.t1_slider.valueChanged.connect(lambda v: self.t1_label.setText(f"{v} ms"))

        # T2 control
        t2_label = QLabel("T2:")
        t2_label.setStyleSheet("color: white;")
        self.t2_slider = QSlider(Qt.Orientation.Horizontal)
        self.t2_slider.setRange(1, 2500)
        self.t2_slider.setValue(250)
        self.t2_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.t2_slider.setTickInterval(250)
        self.t2_label = QLabel("250 ms")
        self.t2_label.setStyleSheet("color: white;")
        self.t2_slider.valueChanged.connect(lambda v: self.t2_label.setText(f"{v} ms"))

        # Temperature control
        temp_label = QLabel("Temperature:")
        temp_label.setStyleSheet("color: white;")
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(4, 500)
        self.temperature_slider.setValue(300)
        self.temperature_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.temperature_slider.setTickInterval(50)
        self.temp_label = QLabel("300 K")
        self.temp_label.setStyleSheet("color: white;")
        self.temperature_slider.valueChanged.connect(
            lambda v: self.temp_label.setText(f"{v} K")
        )

        # Add widgets to grid layout
        input_layout.addWidget(self.more_info_btn, 0, 0)
        input_layout.addWidget(material_label, 0, 1)
        input_layout.addWidget(self.material_selector, 0, 2)
        input_layout.addWidget(qec_label, 0, 3)
        input_layout.addWidget(self.qec_selector, 0, 4)

        input_layout.addWidget(t1_label, 1, 0)
        input_layout.addWidget(self.t1_slider, 1, 1, 1, 2)
        input_layout.addWidget(self.t1_label, 1, 3)

        input_layout.addWidget(t2_label, 2, 0)
        input_layout.addWidget(self.t2_slider, 2, 1, 1, 2)
        input_layout.addWidget(self.t2_label, 2, 3)

        input_layout.addWidget(temp_label, 3, 0)
        input_layout.addWidget(self.temperature_slider, 3, 1, 1, 2)
        input_layout.addWidget(self.temp_label, 3, 3)

        # Run button with reference styling
        self.analyze_btn = QPushButton("Run Analysis")
        self.analyze_btn.setStyleSheet(
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
        self.analyze_btn.clicked.connect(self.run_analysis)
        input_layout.addWidget(
            self.analyze_btn, 4, 0, 1, 5, Qt.AlignmentFlag.AlignCenter
        )

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Create visualization section
        viz_container = QWidget()
        viz_layout = QHBoxLayout(viz_container)

        # Left side: Plots
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)

        # Right side: Results text
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(
            """
            QTextEdit {
                background-color: white;
                border: 1px solid #cccccc;
                padding: 8px;
                font-family: monospace;
                font-size: 12px;
                color: #000000;
            }
        """
        )

        # Add plot and results to visualization container
        viz_layout.addWidget(self.canvas, stretch=2)
        viz_layout.addWidget(self.results_text, stretch=1)

        # Add visualization section to main layout
        layout.addWidget(viz_container)

        # Set the layout
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
