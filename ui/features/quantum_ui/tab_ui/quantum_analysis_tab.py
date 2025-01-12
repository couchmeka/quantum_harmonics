import traceback

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QFileDialog,
    QScrollArea,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit import qasm3
from qiskit.visualization import plot_state_qsphere
import numpy as np
import librosa
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from core.calculations.quantum.quantum_state_display import QuantumStateVisualizer
from data.backend_data_management.data_manager import QuantumDataManager
from ui.styles_ui.styles import create_group_box, create_button, textedit_style


class QuantumAnalysisTab(QWidget):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_manager = QuantumDataManager()

        # Initialize core components
        self.sample_rate = None
        self.audio_data = None
        self.last_results = None

        # Initialize matplotlib figure and canvas
        self.figure = Figure(figsize=(15, 12))
        self.figure.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.figure)

        # Initialize quantum state visualizer
        self.visualizer = QuantumStateVisualizer(self.figure)

        # Initialize UI components
        self.results_text = None
        self.import_btn = None
        self.analyze_btn = None
        self.export_data_btn = None
        self._figure_refs = []
        self.gl_widget = QOpenGLWidget(self)

        # Set up the UI
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 10)
        layout.setSpacing(5)

        # Controls group
        controls_group = create_group_box("Analysis Controls")
        controls_layout = QHBoxLayout()

        # Buttons
        self.import_btn = create_button("Import Audio")
        self.analyze_btn = create_button("Run Analysis")
        self.export_data_btn = create_button("Export Data")

        controls_layout.addWidget(self.import_btn)
        controls_layout.addWidget(self.analyze_btn)
        controls_layout.addWidget(self.export_data_btn)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Main content area
        content_container = QWidget()
        content_layout = QHBoxLayout(content_container)

        # Visualization area
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)

        # Qsphere visualization
        viz_group = create_group_box("Quantum State Analysis")
        viz_group_layout = QVBoxLayout()

        self.canvas.setMinimumHeight(500)
        self.gl_widget.setParent(self)
        viz_group_layout.addWidget(self.gl_widget)
        viz_group_layout.addWidget(self.canvas)

        viz_group.setLayout(viz_group_layout)
        viz_layout.addWidget(viz_group)

        content_layout.addWidget(viz_container, stretch=2)

        # Results area
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet(
            textedit_style + "min-width: 300px; max-width: 400px;"
        )

        scroll.setWidget(self.results_text)
        results_layout.addWidget(scroll)

        content_layout.addWidget(results_container, stretch=1)

        layout.addWidget(content_container)
        self.setLayout(layout)

        # Connect signals
        self.import_btn.clicked.connect(self.import_audio)
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.export_data_btn.clicked.connect(self.export_data)

    def import_audio(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                parent=self,
                caption="Open Audio File",
                directory="",
                filter="Audio Files (*.wav *.mp3 *.m4a);;All Files (*)",
            )

            if file_name:
                self.audio_data, self.sample_rate = librosa.load(
                    file_name, sr=22050, mono=True, duration=30
                )
                self.results_text.append("Audio loaded successfully!")
        except Exception as e:
            print(f"Error in audio import: {str(e)}")
            self.results_text.append(f"Error loading audio: {str(e)}")

    def run_analysis(self):
        try:
            if not hasattr(self, "audio_data") or self.audio_data is None:
                self.results_text.append("Please import an audio file first.")
                return

            # FFT analysis
            window_size = 2048
            frequency_transform = librosa.stft(self.audio_data, n_fft=window_size)
            frequency_spectrum = librosa.fft_frequencies(sr=self.sample_rate)
            magnitude_spectrum = np.abs(frequency_transform)

            # Extract significant frequencies
            average_magnitudes = np.mean(magnitude_spectrum, axis=1)
            significant_indices = np.argsort(average_magnitudes)[-10:]
            dominant_frequencies = frequency_spectrum[significant_indices]
            dominant_amplitudes = average_magnitudes[significant_indices]

            # Create basic quantum circuit for demonstration
            qc = QuantumCircuit(2)
            qc.h([0, 1])
            qc.cz(0, 1)
            qc.ry(np.pi / 3, 0)
            qc.rx(np.pi / 5, 1)
            qc.z(1)

            # Generate density matrix
            density_matrix = DensityMatrix(qc)

            # Calculate purity
            rho = density_matrix.data
            rho_squared = rho @ rho
            purity = np.trace(rho_squared).real

            # Prepare comprehensive quantum results
            quantum_results = {
                "quantum_frequencies": dominant_frequencies.tolist(),
                "amplitudes": dominant_amplitudes.tolist(),
                "noise_levels": {
                    "internal": 0,
                    "external": 0,
                },
                "analysis_results": {
                    "fidelity": 1.0,
                    "phases": [
                        np.angle(f + 1j * a)
                        for f, a in zip(dominant_frequencies, dominant_amplitudes)
                    ],
                },
                "sample_rate": self.sample_rate,
                "density_matrix": density_matrix,
                "statevector": density_matrix.data.flatten().tolist(),
                "purity": purity,
                "fidelity": 1.0,
                "phases": [
                    np.angle(f + 1j * a)
                    for f, a in zip(dominant_frequencies, dominant_amplitudes)
                ],
                "harmony_data": {
                    "notes": [
                        librosa.hz_to_note(freq) for freq in dominant_frequencies
                    ],
                    "frequencies": dominant_frequencies.tolist(),
                    "musical_systems": {},
                },
                "circuit": qc,
                # Add these specific fields for particle simulation
                "particle_data": {
                    "frequencies": dominant_frequencies.tolist(),
                    "amplitudes": dominant_amplitudes.tolist(),
                    "quantum_state": density_matrix.data.flatten().tolist(),
                    "phases": [
                        np.angle(f + 1j * a)
                        for f, a in zip(dominant_frequencies, dominant_amplitudes)
                    ],
                },
            }

            # Update visualization
            self.update_visualization(quantum_results)

            # Print debug info before sending
            print("Sending quantum data to manager:")
            print(
                f"Number of frequencies: {len(quantum_results['quantum_frequencies'])}"
            )
            print(f"Quantum state vector size: {len(quantum_results['statevector'])}")

            # Send data to manager for ML analysis
            self.data_manager.update_quantum_results(quantum_results)

            # Store results and emit signal
            self.last_results = quantum_results
            self.analysis_complete.emit(quantum_results)

            self.results_text.append("Analysis complete! Data sent to ML pipeline.")

        except Exception as e:
            print(f"Analysis error: {str(e)}")
            self.results_text.append(f"Error during analysis: {str(e)}")
            traceback.print_exc()

    def update_visualization(self, results):
        try:
            self.figure.clf()

            # Create single Qsphere visualization
            ax = self.figure.add_subplot(111, projection="3d")
            ax.grid(False)
            ax.set_axis_off()
            plot_state_qsphere(
                results["density_matrix"],
                show_state_phases=True,
                use_degrees=True,
                ax=ax,
            )

            self.figure.tight_layout(pad=2.0)
            self.canvas.draw()

            # Update results text with analysis information
            self._update_results_text(results)

        except Exception as e:
            print(f"Visualization error: {e}")

    def _update_results_text(self, results):
        """Update the results text area with analysis information"""
        self.results_text.clear()
        self.results_text.append("=== Quantum Analysis Results ===\n")

        # Quantum metrics
        self.results_text.append(f"State Purity: {results['purity']:.3f}")
        self.results_text.append(f"State Fidelity: {results['fidelity']:.3f}\n")

        # Frequency analysis
        self.results_text.append("=== Frequency Analysis ===")
        for i, (freq, amp, note) in enumerate(
            zip(
                results["quantum_frequencies"],
                results["amplitudes"],
                results["harmony_data"]["notes"],
            )
        ):
            self.results_text.append(f"\nPeak {i + 1}:")
            self.results_text.append(f"• Frequency: {freq:.1f} Hz")
            self.results_text.append(f"• Amplitude: {amp:.3f}")
            self.results_text.append(f"• Musical Note: {note}")
            self.results_text.append(
                f"• Phase: {np.degrees(results['phases'][i]):.2f}°"
            )

    def export_data(self):
        if not hasattr(self, "last_results"):
            self.results_text.append("No data to export. Run analysis first.")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "JSON Files (*.json)"
        )

        if file_name:
            try:
                import json
                from datetime import datetime

                # Prepare export data
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "quantum_state": {
                        "statevector": self.last_results["statevector"],
                        "circuit_depth": len(self.last_results["circuit"]),
                    },
                    "audio_metadata": self.last_results["audio_data"],
                }

                with open(file_name, "w") as f:
                    json.dump(export_data, f, indent=2)

                self.results_text.append(f"Data exported to {file_name}")
            except Exception as e:
                self.results_text.append(f"Error exporting data: {str(e)}")

    def cleanup(self):
        if hasattr(self, "canvas"):
            self.canvas.close()
        self._figure_refs.clear()
