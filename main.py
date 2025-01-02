# main.py
import sys

import librosa
import numpy as np
from PyQt6.QtWidgets import QApplication, QFileDialog
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure

from storage.data_manager import QuantumDataManager
from ui.features.quantum_ui.tab_ui.fluid_dynamics_tab import FluidAnalysisTab

# Main UI imports
from ui.launch_screen import LaunchScreen
from ui.feature_screen import FeatureScreen

# Quantum Harmonic Analysis
from ui.features.quantum_ui.tab_ui.quantum_analysis_tab import QuantumAnalysisTab
from ui.features.quantum_ui.tab_ui.qec_analysis_tab import QECAnalysisTab
from ui.features.quantum_ui.tab_ui.note_tab import MusicTable
from ui.features.quantum_ui.tab_ui.melody_analysis_tab import MelodyAnalysisTab
from ui.features.quantum_ui.tab_ui.particle_simulation_tab import ParticleSimulationTab
from ui.features.quantum_ui.tab_ui.circuit_tab import QuantumMelodyAnalysisTab
from ui.features.ai_ui.ai_analysis_screen import AIAnalysisScreen

from dotenv import load_dotenv

load_dotenv()


class MainApplication:
    def __init__(self):
        self.qec_tab = None
        self.fluid_tab = None
        self.app = QApplication(sys.argv)
        self.launch_screen = LaunchScreen()
        self.feature_screen = None

        # Initialize audio-related attributes
        self.audio_data = None
        self.sample_rate = None
        self.melody_tab = None
        self.melody_quantum_tab = None

        # Initialize visualization attributes
        self.figure = None
        self.canvas = None

        # Initialize quantum tabs dictionary
        self.quantum_tabs = {
            "Quantum Analysis": QuantumAnalysisTab,
            "Circuit": QuantumMelodyAnalysisTab,
            "Melody Analysis": MelodyAnalysisTab,
            "Note Table": MusicTable,
            "Fluid Dynamics": FluidAnalysisTab,
            "QEC": QECAnalysisTab,
            "Simulator": ParticleSimulationTab,
            "AI": AIAnalysisScreen,
        }

        # Connect the launch screen to feature screen
        self.launch_screen.finished.connect(self.show_feature_screen)

    # In MainApplication class, update the show_feature_screen method

    def show_feature_screen(self):
        self.feature_screen = FeatureScreen(quantum_tabs=self.quantum_tabs)

        # Find all tabs
        self.fluid_tab = self.feature_screen.findChild(FluidAnalysisTab)
        self.qec_tab = self.feature_screen.findChild(QECAnalysisTab)
        quantum_melody_tab = self.feature_screen.findChild(QuantumMelodyAnalysisTab)
        particle_tab = self.feature_screen.findChild(ParticleSimulationTab)
        quantum_analysis_tab = self.feature_screen.findChild(QuantumAnalysisTab)
        melody_analysis_tab = self.feature_screen.findChild(MelodyAnalysisTab)
        ai_analysis_screen = self.feature_screen.findChild(AIAnalysisScreen)

        # Create data manager instance
        data_manager = QuantumDataManager()

        # Connect analysis tabs to update the data manager
        if quantum_analysis_tab:
            quantum_analysis_tab.analysis_complete.connect(
                lambda results: data_manager.update_quantum_results(results)
            )
            print("Connected Quantum Analysis to Data Manager")

        if self.fluid_tab:
            self.fluid_tab.analysis_complete.connect(
                lambda results: data_manager.update_fluid_results(results)
            )
            print("Connected Fluid Analysis to Data Manager")

        if melody_analysis_tab:
            melody_analysis_tab.analysis_complete.connect(
                lambda results: data_manager.update_melody_results(results)
            )
            print("Connected Melody Analysis to Data Manager")

        if quantum_melody_tab:
            quantum_melody_tab.analysis_complete.connect(
                lambda results: data_manager.update_quantum_results(results)
            )
            print("Connected Quantum Melody Analysis to Data Manager")

        if self.qec_tab:
            self.qec_tab.analysis_complete.connect(
                lambda results: data_manager.update_qec_results(results)
            )
            print("Connected QEC Analysis to Data Manager")

        if ai_analysis_screen:
            ai_analysis_screen.analysis_complete.connect(
                lambda results: data_manager.update_all_results(results)
            )
            print("Connected AI Analysis to Data Manager")

        # Connect data flow TO particle simulator
        if particle_tab and quantum_melody_tab:
            quantum_melody_tab.analysis_complete.connect(
                particle_tab.import_quantum_data
            )
            print("Connected Quantum Analysis to Particle Simulator")

        # Connect Fluid to QEC
        if self.fluid_tab and self.qec_tab:
            self.fluid_tab.analysis_complete.connect(self.qec_tab.import_fluid_data)
            print("Connected Fluid Analysis to QEC")

        self.feature_screen.show()
        self.launch_screen.close()

    def run(self):
        self.launch_screen.show()
        return self.app.exec()

    def import_audio(self):
        """Import audio file using QFileDialog"""
        file_name, _ = QFileDialog.getOpenFileName(
            parent=self.feature_screen,  # Specify parent
            caption="Open Audio File",
            directory="",
            filter="Audio Files (*.wav *.mp3 *.m4a);;All Files (*)",
        )

        if file_name:
            try:
                print("Loading audio file...")
                self.audio_data, self.sample_rate = librosa.load(
                    file_name, sr=22050, mono=True, duration=30
                )
                print(
                    f"Audio loaded - Sample rate: {self.sample_rate}, Data shape: {self.audio_data.shape}"
                )

                # Create figure if it doesn't exist
                if self.figure is None:
                    self.figure = Figure(figsize=(10, 5))
                    self.canvas = FigureCanvas(self.figure)

                self.plot_audio()
                print("Plot completed")

                # Update tabs with new audio data if they exist
                if hasattr(self, "melody_tab") and self.melody_tab:
                    self.melody_tab.update_audio_data(self.audio_data, self.sample_rate)
                    print("Melody tab updated")

                if hasattr(self, "melody_quantum_tab") and self.melody_quantum_tab:
                    self.melody_quantum_tab.update_audio_data(
                        self.audio_data, self.sample_rate
                    )
                    print("Quantum tab updated")

            except Exception as e:
                print(f"Error loading audio: {str(e)}")

    def plot_audio(self):
        """Plot audio waveform"""
        if self.audio_data is None or self.figure is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        times = np.arange(len(self.audio_data)) / self.sample_rate
        ax.plot(times, self.audio_data)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Audio Waveform")

        if self.canvas:
            self.canvas.draw()


if __name__ == "__main__":
    main_app = MainApplication()
    sys.exit(main_app.run())
