from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import librosa
import librosa.display
import numpy as np
from tensorflow import keras


class AIQuantumMapper:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        inputs = keras.Input(shape=(2,))
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        outputs = keras.layers.Dense(16, activation='relu')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict_quantum_states(self, frequencies, amplitudes):
        features = np.column_stack((frequencies, amplitudes))
        return self.model.predict(features)


class MelodyQuantumTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data = None
        self.sample_rate = None
        self.ai_mapper = AIQuantumMapper()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_audio_data(self, audio_data, sample_rate):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        if self.audio_data is not None:
            hop_length = 1024
            self.audio_data = librosa.resample(self.audio_data, orig_sr=self.sample_rate, target_sr=22050)
            self.sample_rate = 22050
            self.analyze_melody_quantum()

    def analyze_melody_quantum(self):
        if self.audio_data is None:
            return

        try:
            self.figure.clear()

            ax1 = self.figure.add_subplot(311)
            D = librosa.stft(self.audio_data, n_fft=2048, hop_length=1024)
            phases = np.angle(D)
            img = librosa.display.specshow(
                phases,
                y_axis='log',
                x_axis='time',
                sr=self.sample_rate,
                ax=ax1,
                cmap='hsv'
            )
            ax1.set_title('Quantum Phase Evolution')
            self.figure.colorbar(img, ax=ax1, format='%+2.0f π')

            ax2 = self.figure.add_subplot(312)
            harmonic = librosa.effects.harmonic(y=self.audio_data, margin=4)
            D_harmonic = librosa.amplitude_to_db(np.abs(librosa.stft(harmonic, hop_length=1024)), ref=np.max)
            librosa.display.specshow(
                D_harmonic,
                y_axis='log',
                x_axis='time',
                ax=ax2
            )
            ax2.set_title('Harmonic-Quantum Coupling')

            ax3 = self.figure.add_subplot(313)
            chroma = librosa.feature.chroma_cqt(
                y=self.audio_data,
                sr=self.sample_rate,
                hop_length=1024,
                n_chroma=12
            )
            librosa.display.specshow(
                chroma,
                y_axis='chroma',
                x_axis='time',
                ax=ax3
            )
            ax3.set_title('Note-State Distribution')

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Error in melody quantum analysis: {str(e)}")
