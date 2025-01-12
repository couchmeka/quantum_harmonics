import numpy as np
import traceback
from PyQt6.QtCore import QMutex, QMutexLocker
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from data.backend_data_management.data_manager import QuantumDataManager
from ui.styles_ui.styles import COLORS
import matplotlib.pyplot as plt


class BlochSphereVisualizer:
    def __init__(self, data_manager, figure, canvas):
        if not isinstance(data_manager, QuantumDataManager):
            raise ValueError("data_manager must be an instance of QuantumDataManager")

        self.data_manager = data_manager
        self.fig = figure
        self.canvas = canvas
        self._canvas_mutex = QMutex()
        self.animation_frame = 0  # Initialize frame counter

        # Setup initial view
        self.setup_bloch_sphere()
        print("BlochSphereVisualizer initialized")

    def setup_bloch_sphere(self):
        """Initial setup of Bloch sphere"""
        with QMutexLocker(self._canvas_mutex):
            try:
                # Start with equal superposition state
                initial_state = Statevector([1 / np.sqrt(2), 1 / np.sqrt(2)])

                # Create plot directly in our figure
                self.fig.clear()
                _ = plot_bloch_multivector(
                    initial_state, title="Quantum State Evolution", font_size=14
                )

                # Update display
                self.canvas.draw_idle()
                print("Bloch sphere setup complete")

            except Exception as e:
                print(f"Error in setup_bloch_sphere: {str(e)}")
                traceback.print_exc()

    def update_for_timestep(self, timestep):
        """Update visualization for specific timestep"""
        try:
            if timestep is None:
                timestep = 0

            # Debug: Check what's in the data manager
            if not self.data_manager:
                print("No data manager")
                return False

            if not self.data_manager.quantum_results:
                print("No quantum results in data manager")
                return False

            # Debug: Check the last quantum result
            last_result = self.data_manager.quantum_results[-1]
            print(f"Last quantum result keys: {last_result.keys()}")

            # Get particle simulator
            particle_sim = last_result.get("simulator")
            if not particle_sim:
                print("No particle simulator in quantum results")
                print(f"Available keys: {last_result.keys()}")
                return False

            # Get quantum state
            psi = particle_sim.get_quantum_states()
            if psi is None:
                print("No quantum states from simulator")
                return False
            if len(psi) < 2:
                print(f"Insufficient quantum states: {len(psi)}")
                return False

            # Create statevector for current timestep
            chunk_size = 2
            n_chunks = len(psi) // chunk_size
            chunk_idx = timestep % n_chunks
            start_idx = chunk_idx * chunk_size

            # Get current chunk of state vector
            state_chunk = psi[start_idx : start_idx + chunk_size]
            state_chunk = np.array(state_chunk, dtype=np.complex128)

            # Normalize
            norm = np.linalg.norm(state_chunk)
            if not np.isclose(norm, 0):
                state_chunk = state_chunk / norm

            # Create quantum state
            state = Statevector(state_chunk)

            with QMutexLocker(self._canvas_mutex):
                # Plot in our figure
                self.fig.clear()
                _ = plot_bloch_multivector(
                    state, title=f"Quantum State (t={timestep})", font_size=14
                )

                # Add state information
                state_str = (
                    f"State: |ψ⟩ = {state_chunk[0]:.2f}|0⟩ + {state_chunk[1]:.2f}|1⟩"
                )
                self.fig.text(
                    0.02,
                    0.95,
                    state_str,
                    transform=self.fig.transFigure,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.5),
                )

                # Update display
                self.canvas.draw_idle()
                print(f"Updated visualization for timestep {timestep}")
                return True

        except Exception as e:
            print(f"Error updating Bloch sphere: {str(e)}")
            traceback.print_exc()
            return False

    def update_animation(self, frame):
        """Animation update function"""
        if frame is None:
            frame = 0

        self.animation_frame = frame  # Store current frame
        if self.update_for_timestep(frame):
            return []

        return []  # Return empty list even on failure to continue animation

    def cleanup(self):
        """Clean up resources"""
        try:
            with QMutexLocker(self._canvas_mutex):
                self.animation_frame = 0
                if hasattr(self, "canvas") and self.canvas:
                    self.canvas.draw_idle()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            traceback.print_exc()

    def _add_annotations(self):
        """Add static annotations to the Bloch sphere"""
        self.bloch.add_annotation("|0⟩", (0, 0, 1.2))
        self.bloch.add_annotation("|1⟩", (0, 0, -1.2))
        self.bloch.add_annotation("|+⟩", (1.2, 0, 0))
        self.bloch.add_annotation("|-⟩", (-1.2, 0, 0))
