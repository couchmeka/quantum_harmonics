# Library imports
import librosa
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QSlider, QTextEdit, QGroupBox, QApplication,  # Add QApplication here
                             QFileDialog, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
import qtawesome as qta

# App Imports
from core.calculations.quantum.quantum_harmonic_analysis import QuantumHarmonicsAnalyzer
from core.calculations.quantum.quantum_state_display import QuantumStateVisualizer
from storage.data_manager import QuantumDataManager


class QuantumAnalysisTab(QWidget):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_manager = QuantumDataManager()

        # Initialize analyzer
        self.more_info_btn = None
        self.sample_rate = None
        self.audio_data = None
        self.last_results = None
        self.analyzer = QuantumHarmonicsAnalyzer()

        # Initialize matplotlib figure and canvas
        self.figure = Figure(figsize=(15, 12))
        self.figure.patch.set_facecolor('#ffffff')  # Dark background
        self.canvas = FigureCanvas(self.figure)

        # Initialize quantum state visualizer with the figure
        self.visualizer = QuantumStateVisualizer(self.figure)

        # Initialize UI components
        self.internal_noise_slider = None
        self.external_noise_slider = None
        self.internal_noise_label = None
        self.external_noise_label = None
        self.results_text = None
        self.import_btn = None
        self.analyze_btn = None
        self.export_data_btn = None
        self.export_plot_btn = None

        # Set up the UI
        self.setup_ui()

    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Controls group
        controls_group = QGroupBox("Analysis Controls")
        controls_group.setStyleSheet("""
            QGroupBox {
                color: white;
                font-size: 14px;
                border: 1px solid #3d405e;
                border-radius: 5px;
                padding: 15px;
                background-color: rgba(61, 64, 94, 0.3);
            }
        """)
        controls_group.setMaximumHeight(160)
        controls_layout = QVBoxLayout()

        button_style = """
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

        # Add noise controls
        noise_layout = QHBoxLayout()

        # Internal noise
        internal_layout = QHBoxLayout()
        internal_label = QLabel("Internal Noise (T1/T2):")
        internal_label.setStyleSheet("color: white;")
        self.internal_noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.internal_noise_slider.setRange(0, 100)
        self.internal_noise_slider.setValue(0)
        self.internal_noise_label = QLabel("0%")
        self.internal_noise_label.setStyleSheet("color: white;")
        internal_layout.addWidget(internal_label)
        internal_layout.addWidget(self.internal_noise_slider)
        internal_layout.addWidget(self.internal_noise_label)
        noise_layout.addLayout(internal_layout)

        # External noise
        external_layout = QHBoxLayout()
        external_label = QLabel("External Noise:")
        external_label.setStyleSheet("color: white;")
        self.external_noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.external_noise_slider.setRange(0, 100)
        self.external_noise_slider.setValue(0)
        self.external_noise_label = QLabel("0%")
        self.external_noise_label.setStyleSheet("color: white;")
        external_layout.addWidget(external_label)
        external_layout.addWidget(self.external_noise_slider)
        external_layout.addWidget(self.external_noise_label)
        noise_layout.addLayout(external_layout)

        controls_layout.addLayout(noise_layout)

        # Button layout
        button_layout = QHBoxLayout()

        # Import button
        self.import_btn = QPushButton("Import Audio")
        self.import_btn.setStyleSheet(button_style)
        self.import_btn.clicked.connect(self.import_audio)

        # Analyze button
        self.analyze_btn = QPushButton("Run Analysis")
        self.analyze_btn.setStyleSheet(button_style)
        self.analyze_btn.clicked.connect(self.run_analysis)

        # Export buttons
        self.export_data_btn = QPushButton("Export Data")
        self.export_data_btn.setStyleSheet(button_style)
        self.export_data_btn.clicked.connect(self.export_data)

        self.export_plot_btn = QPushButton("Export Plot")
        self.export_plot_btn.setStyleSheet(button_style)
        self.export_plot_btn.clicked.connect(self.export_plot)

        # Create a More Info button
        self.more_info_btn = QPushButton()
        self.more_info_btn.setIcon(qta.icon('fa.question-circle', color='#2196F3'))  # Set blue color
        self.more_info_btn.setStyleSheet("""
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
                """)
        self.more_info_btn.setToolTip("More information about the plots")
        self.more_info_btn.clicked.connect(self.show_plot_info)

        # Add buttons
        button_layout.addWidget(self.import_btn)
        button_layout.addWidget(self.analyze_btn)
        button_layout.addWidget(self.export_plot_btn)
        button_layout.addWidget(self.export_data_btn)
        button_layout.addWidget(self.more_info_btn)  # Add this line

        controls_layout.addLayout(button_layout)
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)

        # Add matplotlib canvas for visualizations
        main_layout.addWidget(self.canvas)

        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                padding: 8px;
                font-family: Arial, sans-serif;
                font-size: 12px;
                color: #000000;
            }
        """)
        self.results_text.setMaximumHeight(100)
        main_layout.addWidget(self.results_text)

        # Connect slider value changes to label updates
        self.internal_noise_slider.valueChanged.connect(
            lambda v: self.internal_noise_label.setText(f"{v}%")
        )
        self.external_noise_slider.valueChanged.connect(
            lambda v: self.external_noise_label.setText(f"{v}%")
        )

    def import_audio(self):
        """Import audio file using QFileDialog"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                parent=self,  # Make sure parent is set
                caption="Open Audio File",
                directory="",
                filter="Audio Files (*.wav *.mp3 *.m4a);;All Files (*)"
            )

            if file_name:
                print("Loading audio file...")
                # Disable buttons during load
                self.import_btn.setEnabled(False)
                self.analyze_btn.setEnabled(False)
                self.export_data_btn.setEnabled(False)
                self.export_plot_btn.setEnabled(False)
                QApplication.processEvents()  # Allow UI to update

                # Load audio
                self.audio_data, self.sample_rate = librosa.load(file_name, sr=22050, mono=True, duration=30)

                # Plot audio waveform
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                times = np.arange(len(self.audio_data)) / self.sample_rate
                ax.plot(times, self.audio_data)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.set_title('Audio Waveform')
                self.canvas.draw()

                # Re-enable buttons
                self.import_btn.setEnabled(True)
                self.analyze_btn.setEnabled(True)
                self.export_data_btn.setEnabled(True)
                self.export_plot_btn.setEnabled(True)

                self.results_text.append("Audio loaded successfully!")

        except Exception as e:
            print(f"Error in audio import: {str(e)}")
            self.results_text.append(f"Error loading audio: {str(e)}")
            # Make sure buttons are re-enabled even if there's an error
            if hasattr(self, 'import_btn'):
                self.import_btn.setEnabled(True)
            if hasattr(self, 'analyze_btn'):
                self.analyze_btn.setEnabled(True)
            if hasattr(self, 'export_data_btn'):
                self.export_data_btn.setEnabled(True)
            if hasattr(self, 'export_plot_btn'):
                self.export_plot_btn.setEnabled(True)

    def run_analysis(self):
        try:
            if not hasattr(self, 'audio_data') or self.audio_data is None:
                self.results_text.append("Please import audio file first.")
                return

            internal_noise = self.internal_noise_slider.value() / 100.0
            external_noise = self.external_noise_slider.value() / 100.0

            # FFT analysis
            window_size = 2048
            frequency_transform = librosa.stft(self.audio_data, n_fft=window_size)
            frequency_spectrum = librosa.fft_frequencies(sr=self.sample_rate)
            magnitude_spectrum = np.abs(frequency_transform)

            average_magnitudes = np.mean(magnitude_spectrum, axis=1)
            significant_indices = np.argsort(average_magnitudes)[-10:]
            dominant_frequencies = frequency_spectrum[significant_indices]
            dominant_amplitudes = average_magnitudes[significant_indices]

            results = self.analyzer.analyze_harmonics(
                frequencies=dominant_frequencies,
                amplitudes=dominant_amplitudes,
                internal_noise=internal_noise,
                external_noise=external_noise
            )

            # Store in data manager
            quantum_results = {
                'frequencies': dominant_frequencies.tolist(),
                'amplitudes': dominant_amplitudes.tolist(),
                'noise_levels': {'internal': internal_noise, 'external': external_noise},
                'analysis_results': results,
                'sample_rate': self.sample_rate,
                'statevector': results.get('statevector', []),  # Add statevector
                'purity': results.get('purity', 0),
                'fidelity': results.get('fidelity', 0)
            }
            self.data_manager.update_quantum_results(quantum_results)

            self.last_results = results
            self.visualizer.plot_quantum_state(results)
            self.canvas.draw()
            self._update_results_text(results, dominant_frequencies, dominant_amplitudes)

        except Exception as e:
            print(f"Analysis error: {str(e)}")
            self.results_text.append(f"Error during analysis: {str(e)}")

    def _update_results_text(self, results, frequencies, amplitudes):
        """Helper method to update results text"""
        self.results_text.clear()
        self.results_text.append("=== Frequency Analysis ===\n")

        for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
            self.results_text.append(f"\nPeak {i + 1}:")
            self.results_text.append(f"• Frequency: {freq:.1f} Hz")
            self.results_text.append(f"• Relative Amplitude: {amp:.3f}")

            if 'musical_mappings' in results:
                for system, mapping in results['musical_mappings'].items():
                    self.results_text.append(
                        f"• {system}: {mapping['note']} (deviation: {mapping['deviation']:.2f} Hz)")

        self._append_quantum_analysis_results(results)

    def quantum_fourier_analysis(self, audio_signal, sample_rate):
        """
        Perform Quantum Fourier Transform analysis on audio signal.

        Args:
            audio_signal: The time-domain audio signal
            sample_rate: Number of samples per second (Hz)

        Returns:
            Quantum Fourier Transform of the signal
        """
        # Get required number of qubits
        num_samples = len(audio_signal)
        num_qubits = len(bin(num_samples)[2:])  # number of qubits needed

        # Create quantum circuit for QFT
        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits)
        circuit = QuantumCircuit(qr, cr)

        # Normalize and encode audio data into quantum state
        normalized_signal = audio_signal / np.max(np.abs(audio_signal))
        for i, amplitude in enumerate(normalized_signal[:2 ** num_qubits]):
            angle = np.arccos(amplitude) if amplitude >= 0 else -np.arccos(abs(amplitude))
            if i < 2 ** num_qubits:
                circuit.ry(angle, qr[i % num_qubits])

        # Apply QFT
        circuit.append(QFT(num_qubits, do_swaps=False), qr)
        circuit.measure(qr, cr)

        # Run on simulator
        simulator = AerSimulator()
        transpiled_circuit = transpile(circuit, simulator)
        job = simulator.run(transpiled_circuit, shots=1024)
        result = job.result()

        # Convert results to frequency domain
        counts = result.get_counts()
        frequencies = []
        amplitudes = []

        for state, count in counts.items():
            freq_index = int(state, 2)
            frequency = freq_index * (sample_rate / (2 ** num_qubits))
            amplitude = count / 1024  # Normalize by number of shots
            frequencies.append(frequency)
            amplitudes.append(amplitude)

        return np.array(amplitudes)

    # the explainer message
    def show_plot_info(self):
        QMessageBox.information(
            self,
            "Plot Information",
            "The visualizations include:\n"
            "1. Quantum Circuit: Shows the quantum operations and gates used in the analysis.\n"
            "2. Quantum Surface: Displays the surface with or without Pythagorean results.\n"
            "3. State Probabilities: Illustrates the probability distribution of states.\n"
            "4. Quantum Quality Metrics: Shows state purity and fidelity measurements."
        )

    def update_internal_noise(self, value):
        """Update internal noise label"""
        self.internal_noise_label.setText(f"{value}%")

    def update_external_noise(self, value):
        """Update external noise label"""
        self.external_noise_label.setText(f"{value}%")

    def export_data(self):
        """Export analysis results"""
        if not hasattr(self, 'last_results'):
            self.results_text.append("\nNo data to export. Run analysis first.")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            "",
            "JSON Files (*.json);;CSV Files (*.csv)"
        )

        if file_name:
            try:
                if file_name.endswith('.json'):
                    self._export_json(file_name)
                elif file_name.endswith('.csv'):
                    self._export_csv(file_name)
                self.results_text.append(f"\nData exported to {file_name}")
            except Exception as e:
                self.results_text.append(f"\nError exporting data: {str(e)}")

    def _export_json(self, filename):
        """Export data as JSON"""
        from datetime import datetime
        import json

        export_data = {
            'timestamp': datetime.now().isoformat(),
            'noise_levels': {
                'internal': self.internal_noise_slider.value() / 100.0,
                'external': self.external_noise_slider.value() / 100.0
            },
            'measurements': dict(self.last_results['counts']),
            'metrics': {
                'purity': float(self.last_results['purity']),
                'fidelity': float(self.last_results['fidelity'])
            },
            'materials_analysis': self.last_results['materials_analysis']
        }

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

    def _export_csv(self, filename):
        """Export data as CSV"""
        from datetime import datetime
        import csv

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write metadata
            writer.writerow(['Analysis Timestamp', datetime.now().isoformat()])
            writer.writerow([''])

            # Write noise levels
            writer.writerow(['Noise Levels'])
            writer.writerow(['Internal Noise', self.internal_noise_slider.value() / 100.0])
            writer.writerow(['External Noise', self.external_noise_slider.value() / 100.0])
            writer.writerow([''])

            # Write measurements
            writer.writerow(['State', 'Count'])
            for state, count in self.last_results['counts'].items():
                writer.writerow([state, count])
            writer.writerow([''])

            # Write metrics
            writer.writerow(['Metrics'])
            writer.writerow(['Purity', float(self.last_results['purity'])])
            writer.writerow(['Fidelity', float(self.last_results['fidelity'])])
            writer.writerow([''])

            # Write materials analysis
            writer.writerow(['Recommended Materials'])
            writer.writerow(['Material', 'Score', 'Coherence Time', 'Coupling Strength', 'Quantum Efficiency'])
            for material in self.last_results['materials_analysis'][:3]:  # Top 3 materials
                writer.writerow([
                    material['material'],
                    material['score'],
                    material['coherence_time'],
                    material['coupling_strength'],
                    material['quantum_efficiency']
                ])

    def export_plot(self):
        """Export current visualization as image"""
        if not hasattr(self, 'last_results'):
            self.results_text.append("\nNo plot to export. Run analysis first.")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf)"
        )

        if file_name:
            try:
                self.figure.savefig(file_name, dpi=300, bbox_inches='tight')
                self.results_text.append(f"\nPlot saved to {file_name}")
            except Exception as e:
                self.results_text.append(f"\nError saving plot: {str(e)}")

    def _append_quantum_analysis_results(self, results):
        """Add quantum analysis results to the text display"""
        if not results:
            return

        self.results_text.append("\n=== Quantum Analysis ===")

        # Add quantum state metrics if available
        if 'purity' in results:
            self.results_text.append(f"\nState Purity: {results['purity']:.3f}")
        if 'fidelity' in results:
            self.results_text.append(f"State Fidelity: {results['fidelity']:.3f}")

        # Add decoherence info if available
        if 'decoherence' in results:
            self.results_text.append(f"Decoherence Time: {results['decoherence']:.2e} s")

        # Add any other quantum-specific results
        if 'statevector' in results:
            self.results_text.append("\nQuantum State Vector:")
            statevector = results['statevector']
            if isinstance(statevector, (list, np.ndarray)):
                for i, amp in enumerate(statevector[:5]):  # Show first 5 amplitudes
                    self.results_text.append(f"State |{i}⟩: {amp:.3f}")
                if len(statevector) > 5:
                    self.results_text.append("...")

        # Add any other quantum metrics you want to display
