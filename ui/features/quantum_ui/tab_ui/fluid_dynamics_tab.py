import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QGroupBox,
    QPushButton,
    QSlider,
    QComboBox,
    QDialog,
    QScrollArea,
)
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from data.frequencies import frequency_systems
from data.elements import atomic_frequencies
from core.calculations.navier.fluid_analysis import FluidAnalyzer
import qtawesome as qta

from storage.data_manager import QuantumDataManager


class FluidDataWorker(QObject):
    finished = pyqtSignal(dict)  # Signal emitted when analysis is complete
    error = pyqtSignal(str)

    def __init__(self, analyzer, notes, re_value, ma_value, pythagorean_ratio):
        super().__init__()
        self.analyzer = analyzer
        self.notes = notes
        self.re_value = re_value
        self.ma_value = ma_value
        self.pythagorean_ratio = pythagorean_ratio
        self.data_manager = QuantumDataManager()

    def run(self):
        try:
            results = self.analyzer.analyze_fluid_dynamics(
                self.notes, self.re_value, self.ma_value, self.pythagorean_ratio
            )
            formatted_results = {
                "original_frequencies": results["original_frequencies"],
                "fibonacci_frequencies": results["fibonacci_frequencies"],
                "solution": results["solution"].tolist(),  # Convert numpy arrays
                "t": results["t"].tolist(),
            }
            self.finished.emit(formatted_results)
        except Exception as e:
            self.error.emit(str(e))


class FluidAnalysisTab(QWidget):
    analysis_complete = pyqtSignal(dict)  # Signal declaration at class level

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_manager = QuantumDataManager()
        self.more_info_btn = None
        self.colorbar = None
        self.worker = None
        self.worker_thread = None
        self.melody_combo = None
        self.system_combo = None
        self.current_results = None
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ratio_slider = None
        self.run_button = None
        self.ma_slider = None
        self.note_input = None
        self.re_slider = None
        self.analyzer = FluidAnalyzer(frequency_systems, atomic_frequencies)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()  # Main layout for the widget/window

        # Create a More Info button
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

        # Input controls group box
        input_group = QGroupBox("Fluid Analysis Controls")
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

        # System selection
        system_layout = QHBoxLayout()
        self.system_combo = QComboBox()
        self.system_combo.addItems(
            ["Western 12-Tone", "Indian Classical", "Arabic", "Gamelan", "Pythagorean"]
        )
        system_label = QLabel("System:")
        system_label.setStyleSheet("color: white;")
        system_layout.addWidget(self.more_info_btn)
        system_layout.addWidget(system_label)
        system_layout.addWidget(self.system_combo)

        # Note input with styling (keeping your original implementation)
        note_layout = QHBoxLayout()
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
        label = QLabel("Notes:")
        label.setStyleSheet("color: white;")
        note_layout.addWidget(label)
        note_layout.addWidget(self.note_input)

        # Reynolds number slider
        re_layout = QHBoxLayout()
        re_label = QLabel("Reynolds Number:")
        re_label.setStyleSheet("color: white;")
        self.re_slider = QSlider(Qt.Orientation.Horizontal)
        self.re_slider.setMinimum(1)
        self.re_slider.setMaximum(1000)
        self.re_slider.setValue(100)
        self.re_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.re_slider.setTickInterval(10)
        self.re_slider.setStyleSheet("color: white;")
        re_layout.addWidget(re_label)
        re_layout.addWidget(self.re_slider)

        # Mach number slider
        ma_layout = QHBoxLayout()
        ma_label = QLabel("Mach Number (x100):")
        ma_label.setStyleSheet("color: white;")
        self.ma_slider = QSlider(Qt.Orientation.Horizontal)
        self.ma_slider.setMinimum(1)
        self.ma_slider.setMaximum(100)
        self.ma_slider.setValue(10)
        self.ma_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ma_slider.setTickInterval(5)
        ma_layout.addWidget(ma_label)
        ma_layout.addWidget(self.ma_slider)

        # Pythagorean ratio slider
        ratio_layout = QHBoxLayout()
        ratio_label = QLabel("Pythagorean Ratio (x10):")
        ratio_label.setStyleSheet("color: white;")
        self.ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.ratio_slider.setMinimum(1)
        self.ratio_slider.setMaximum(100)
        self.ratio_slider.setValue(10)
        self.ratio_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.ratio_slider.setTickInterval(5)
        ratio_layout.addWidget(ratio_label)
        ratio_layout.addWidget(self.ratio_slider)

        # Run button
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setStyleSheet(
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

        # Add widget and buttons to input layout
        input_layout.addLayout(system_layout)
        input_layout.addLayout(note_layout)
        input_layout.addLayout(re_layout)
        input_layout.addLayout(ma_layout)
        input_layout.addLayout(ratio_layout)
        input_layout.addWidget(self.run_button)
        input_group.setLayout(input_layout)

        # Add group box to main layout
        layout.addWidget(input_group)

        # Add group box to main layout
        layout.addWidget(self.canvas)

        # Set layout to parent widget
        self.setLayout(layout)

        # Connect button to a slot (optional)
        self.run_button.clicked.connect(self.run_analysis)
        self.run_button.clicked.connect(self.start_analysis)

        # Connect the system combo box to update_note_placeholder
        self.system_combo.currentTextChanged.connect(self.update_note_placeholder)

        # Initial update of note placeholder
        self.update_note_placeholder(self.system_combo.currentText())

    def start_analysis(self):
        notes = self.note_input.text().split(",")
        Re = self.re_slider.value()
        Ma = self.ma_slider.value() / 100.0
        pythagorean_ratio = self.ratio_slider.value() / 10.0

        # Set up the worker thread
        self.worker_thread = QThread()
        self.worker = FluidDataWorker(self.analyzer, notes, Re, Ma, pythagorean_ratio)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Start the thread
        self.worker_thread.start()

    def on_analysis_complete(self, results):
        self.worker_thread.quit()
        self.current_results = results
        self.analysis_complete.emit(results)
        self.plot_results(results)

    def on_analysis_error(self, error_message):
        self.worker_thread.quit()
        print(f"Error during analysis: {error_message}")

    def run_analysis(self):  # In FluidAnalysisTab
        try:
            print("\nStarting Fluid Analysis...")
            notes = self.note_input.text().split(",")
            Re = self.re_slider.value()
            Ma = self.ma_slider.value() / 100.0
            pythagorean_ratio = self.ratio_slider.value() / 10.0

            print(f"Analysis parameters:")
            print(f"Notes: {notes}")
            print(f"Reynolds number: {Re}")
            print(f"Mach number: {Ma}")
            print(f"Pythagorean ratio: {pythagorean_ratio}")

            results = self.analyzer.analyze_fluid_dynamics(
                notes, Re, Ma, pythagorean_ratio
            )

            # Log results structure
            print("\nFluid Analysis Results:")
            print(f"Original frequencies: {len(results['original_frequencies'])} items")
            print(
                f"Fibonacci frequencies: {len(results['fibonacci_frequencies'])} items"
            )
            print(f"Solution shape: {np.array(results['solution']).shape}")
            print(f"Time points: {len(results['t'])} items")

            # Format results properly before emitting
            formatted_results = {
                "original_frequencies": results["original_frequencies"],
                "fibonacci_frequencies": results["fibonacci_frequencies"],
                "solution": (
                    results["solution"].tolist()
                    if isinstance(results["solution"], np.ndarray)
                    else results["solution"]
                ),
                "t": (
                    results["t"].tolist()
                    if isinstance(results["t"], np.ndarray)
                    else results["t"]
                ),
            }

            print("\nFormatted results structure:")
            for key, value in formatted_results.items():
                if isinstance(value, list):
                    print(f"{key}: {len(value)} items")
                else:
                    print(f"{key}: {type(value)}")

            print("Analysis completed. Updating data manager...")
            self.data_manager.update_fluid_results(formatted_results)
            print("Data manager updated")

            self.current_results = formatted_results
            print("Emitting analysis_complete signal")
            self.analysis_complete.emit(formatted_results)
            print("Plotting results")
            self.plot_results(results)

        except Exception as e:
            print(f"Analysis error: {str(e)}")
            import traceback

            traceback.print_exc()

    def plot_results(self, results):
        # Add a new color bar for this plot if needed
        if hasattr(self, "colorbar") and self.colorbar is not None:
            try:
                self.colorbar.remove()  # Remove old colorbar
                self.colorbar = None  # Reset to avoid further issues
            except KeyError:
                pass  # Ignore if it doesn't exist
        self.figure.clear()

        gs = self.figure.add_gridspec(2, 3, hspace=0.6, wspace=0.3)

        solution = np.array(results["solution"])

        # Velocity Components
        ax1 = self.figure.add_subplot(gs[0, 0])
        ax1.plot(results["t"], solution[:, 0:3], color="green")
        ax1.set_title("Velocity Components")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Velocity (m/s)")

        # Create a dummy ScalarMappable for the color bar
        sm = plt.cm.ScalarMappable(
            cmap="viridis", norm=plt.Normalize(vmin=solution.min(), vmax=solution.max())
        )
        sm.set_array([])
        self.colorbar = self.figure.colorbar(sm, ax=ax1, orientation="vertical")
        self.colorbar.set_label("Velocity Scale")

        # Pressure Evolution
        ax2 = self.figure.add_subplot(gs[0, 1])
        ax2.plot(results["t"], solution[:, 3], color="blue")
        ax2.set_title("Quantum Pressure Field")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Pressure")

        # Energy Spectrum
        ax3 = self.figure.add_subplot(gs[0, 2])
        v_mag = np.sqrt(np.sum(solution[:, 0:3] ** 2, axis=1))
        ax3.semilogy(results["t"], v_mag, color="#0099ff")
        ax3.set_title("Velocity Magnitude")
        ax3.set_xlabel("Tunneling Probability")
        ax3.set_ylabel("|V| (m/s)")

        # Bottom row - New visualizations

        # Frequency Comparison with Fibonacci Components
        ax4 = self.figure.add_subplot(gs[1, 0])
        musical_freqs = np.array(results["original_frequencies"])
        fibonacci_freqs = np.array(results["fibonacci_frequencies"])
        atomic_freqs = musical_freqs * 1e12  # Convert to THz
        quantum_states = musical_freqs * 1e6  # Convert to MHz

        ax4.scatter(
            musical_freqs, [1] * len(musical_freqs), label="Musical (Hz)", color="green"
        )
        ax4.scatter(
            atomic_freqs, [2] * len(atomic_freqs), label="Atomic (THz)", color="blue"
        )
        ax4.scatter(
            quantum_states,
            [3] * len(quantum_states),
            label="Quantum (MHz)",
            color="purple",
        )
        ax4.scatter(
            fibonacci_freqs,
            [4] * len(fibonacci_freqs),
            label="Fibonacci (Hz)",
            color="orange",
        )

        ax4.set_xscale("log")
        ax4.set_yticks([1, 2, 3, 4])
        ax4.set_yticklabels(["Musical", "Atomic", "Quantum", "Fibonacci"])
        ax4.set_xlabel("Frequency (log scale)")
        ax4.set_title("Frequency Comparison")
        # Move legend to bottom right for better visibility
        ax4.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0))

        # Terahertz Transitions
        ax5 = self.figure.add_subplot(gs[1, 1], projection="3d")
        x = y = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-(X**2 + Y**2))  # Gaussian distribution
        surf = ax5.plot_surface(X, Y, Z, cmap="viridis")
        ax5.set_title("Terahertz Transitions")
        colorbar = self.figure.colorbar(surf, ax=ax5, label="Transition Intensity")

        # Quantum-Classical Resonance
        ax6 = self.figure.add_subplot(gs[1, 2])
        freqs = np.linspace(0, 4e13, 1000)
        resonance = np.sin(2 * np.pi * freqs / 1e13) ** 2
        ax6.plot(freqs, resonance, "b-", label="Resonance Strength")
        for f in atomic_freqs.flatten():
            ax6.axvline(f, color="red", linestyle="--", alpha=0.5)
        ax6.set_title("Quantum-Classical Resonance")
        ax6.legend()

        # Set common styling
        for ax in [ax1, ax2, ax3, ax4, ax6]:
            ax.grid(True, linestyle="--", alpha=0.2)
            ax.set_facecolor("#f8f9fa")

        self.canvas.draw()

    def get_current_results(self):
        if hasattr(self, "current_results"):
            return self.current_results
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
