from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa
import librosa.display
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator


class MelodyMappingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize attributes
        self.audio_data = None
        self.sample_rate = None
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)

        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Add placeholder text
        placeholder = QLabel("Import audio in Quantum Analysis tab to see mapping")
        placeholder.setStyleSheet("color: white; font-size: 14px;")
        layout.addWidget(placeholder, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add canvas
        layout.addWidget(self.canvas)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(12, 14))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    @staticmethod
    def quantum_fourier_analysis(audio_signal, sample_rate):
        """
        Perform Quantum Fourier Transform analysis on an audio signal.

        Args:
            audio_signal: The time-domain audio signal
            sample_rate: Number of samples per second (Hz)

        Returns:
            frequencies: Array of positive frequencies in Hz
            amplitudes: Magnitude of each frequency component
            phases: Phase angle of each frequency component in radians
            dc_offset: Mean value of the signal
        """
        # Normalize and prepare the signal for quantum processing
        num_qubits = len(bin(len(audio_signal))[2:])  # Number of qubits needed
        normalized_signal = audio_signal / np.max(np.abs(audio_signal))  # Normalize to [-1,1]

        # Create quantum registers and circuit
        qr = QuantumRegister(num_qubits, 'q')
        cr = ClassicalRegister(num_qubits, 'c')
        qc = QuantumCircuit(qr, cr)

        # Initialize circuit with signal data
        for i, amp in enumerate(normalized_signal[:2 ** num_qubits]):
            # Convert amplitude to rotation angle
            theta = np.arccos(amp) if amp >= 0 else -np.arccos(abs(amp))
            if i < 2 ** num_qubits:
                qc.ry(theta, qr[i])

        # Apply QFT
        qc.append(QFT(num_qubits), qr)

        # Measure the circuit
        qc.measure(qr, cr)

        # Set up simulator and run circuit
        simulator = AerSimulator()
        transpiled_circuit = transpile(qc, simulator)
        job = simulator.run(transpiled_circuit, shots=1024)
        result = job.result()
        counts = result.get_counts()

        # Process results
        frequencies = []
        amplitudes = []
        phases = []

        # Convert quantum measurements to frequency domain
        for state, count in counts.items():
            freq_index = int(state, 2)
            frequency = freq_index * (sample_rate / (2 ** num_qubits))
            amplitude = count / 1024  # Normalize by number of shots
            phase = np.angle(complex(amplitude))

            frequencies.append(frequency)
            amplitudes.append(amplitude)
            phases.append(phase)

        # Calculate DC offset
        dc_offset = np.mean(audio_signal)

        # Sort by frequency
        sorted_indices = np.argsort(frequencies)
        frequencies = np.array(frequencies)[sorted_indices]
        amplitudes = np.array(amplitudes)[sorted_indices]
        phases = np.array(phases)[sorted_indices]

        return frequencies, amplitudes, phases, dc_offset

    def update_audio_data(self, audio_data, sample_rate):
        if audio_data is None or sample_rate is None:
            return

        self.audio_data = audio_data
        self.sample_rate = sample_rate

        try:
            self.analyze_melody()
        except Exception as e:
            print(f"Error in melody analysis: {str(e)}")

    def analyze_melody(self):
        if self.audio_data is None or self.sample_rate is None:
            return

        try:
            self.figure.clear()

            # Create time plot
            ax1 = self.figure.add_subplot(311)
            times = np.arange(len(self.audio_data)) / self.sample_rate
            ax1.plot(times, self.audio_data)
            ax1.set_title('Waveform')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')

            # Create frequency plot
            # Create frequency plot using Short-Time Fourier Transform
            frequency_transform = librosa.stft(self.audio_data)
            spectrogram_db = librosa.amplitude_to_db(np.abs(frequency_transform), ref=np.max)

            ax2 = self.figure.add_subplot(312)
            librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='hz', ax=ax2)
            ax2.set_title('Frequency Analysis')

            # Create pitch detection
            pitches, magnitudes = librosa.piptrack(y=self.audio_data, sr=self.sample_rate)
            ax3 = self.figure.add_subplot(313)
            librosa.display.specshow(pitches, x_axis='time', y_axis='hz', ax=ax3)
            ax3.set_title('Pitch Tracking')

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Error in melody visualization: {str(e)}")
