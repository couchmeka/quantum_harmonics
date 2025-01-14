from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QMutex, QMutexLocker
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector
import numpy as np


class BlochSphereManager:
    def __init__(self, parent_widget, data_manager):
        self.canvas = None
        self.parent = parent_widget
        self.data_manager = data_manager
        self._draw_mutex = QMutex()
        self._data_mutex = QMutex()
        self.current_state = None
        self.current_metrics = {}
        self._is_cleanup = False
        self._setup_visualization()

    def _setup_visualization(self):
        """Initialize visualization components"""
        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setParent(self.parent)

        # Keep strong reference
        self._canvas_ref = self.canvas
        self._figure_ref = self.figure

        # Configure initial view
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.grid(False)
        self.ax.set_axis_off()

    def fetch_latest_quantum_state(self):
        """Fetch latest quantum state from data manager"""
        with QMutexLocker(self._data_mutex):
            latest_results = self.data_manager.get_latest_quantum_results()
            if not latest_results:
                return None, {}

            state_vector = latest_results.get("statevector")
            metrics = {
                "purity": latest_results.get("purity", 0.0),
                "fidelity": latest_results.get("fidelity", 0.0),
                "frequencies": latest_results.get("quantum_frequencies", []),
            }
            return state_vector, metrics

    def process_state_vector(self, state_vector):
        """Process state vector into proper format"""
        try:
            if isinstance(state_vector, list):
                # Handle nested lists
                if isinstance(state_vector[0], list):
                    state_vector = [
                        item for sublist in state_vector for item in sublist
                    ]
                state_vector = np.array(state_vector, dtype=complex)

            # Take top two amplitudes for qubit visualization
            magnitudes = np.abs(state_vector)
            top_indices = np.argsort(magnitudes)[-2:]
            state_2d = state_vector[top_indices]

            # Normalize
            norm = np.linalg.norm(state_2d)
            if norm > 0:
                state_2d = state_2d / norm

            return Statevector(state_2d)

        except Exception as e:
            print(f"Error processing state vector: {e}")
            return None

    def update_state(self, state_vector=None, frequencies=None):
        """Update visualization with new quantum state"""
        with QMutexLocker(self._draw_mutex):
            try:
                if state_vector is None:
                    state_vector, metrics = self.fetch_latest_quantum_state()
                    if state_vector is None:
                        return False
                    self.current_metrics = metrics

                # Process state vector
                processed_state = self.process_state_vector(state_vector)
                if processed_state is None:
                    return False

                self.current_state = processed_state

                # Update visualization
                self.figure.clear()
                ax = self.figure.add_subplot(111, projection="3d")

                # Generate Bloch sphere
                bloch_fig = plot_bloch_multivector(processed_state)
                if not bloch_fig or not bloch_fig.get_axes():
                    return False

                # Copy visualization to our figure
                qiskit_ax = bloch_fig.get_axes()[0]

                # Copy plot properties
                ax.set_title("Quantum State Evolution")
                ax.set_xlabel(qiskit_ax.get_xlabel())
                ax.set_ylabel(qiskit_ax.get_ylabel())
                ax.set_zlabel(qiskit_ax.get_zlabel())

                # Copy view limits and angle
                ax.set_xlim(qiskit_ax.get_xlim())
                ax.set_ylim(qiskit_ax.get_ylim())
                ax.set_zlim(qiskit_ax.get_zlim())
                ax.view_init(elev=qiskit_ax.elev, azim=qiskit_ax.azim)

                # Add state information
                state_text = f"|ψ⟩ = ({processed_state[0]:.2f})|0⟩ + ({processed_state[1]:.2f})|1⟩"
                metrics_text = f"\nPurity: {self.current_metrics.get('purity', 0):.3f}"
                if frequencies or self.current_metrics.get("frequencies"):
                    freq = (
                        frequencies[0]
                        if frequencies
                        else self.current_metrics["frequencies"][0]
                    )
                    metrics_text += f"\nFrequency: {freq:.2f} Hz"

                self.figure.text(
                    0.1,
                    0.95,
                    state_text + metrics_text,
                    transform=self.figure.transFigure,
                    backgroundcolor="black",
                    color="white",
                )

                plt.close(bloch_fig)  # Clean up temporary figure
                self.safe_draw()
                return True

            except Exception as e:
                print(f"Error updating Bloch sphere: {e}")
                return False

    def safe_draw(self):
        """Safely draw the canvas with error handling"""
        try:
            if self.canvas and not self.canvas.isHidden():
                self.canvas.draw()
                QApplication.processEvents()  # Ensure drawing completes
        except RuntimeError as e:
            print(f"Draw error: {e}")

    def cleanup(self):
        if self._is_cleanup:  # Prevent double cleanup
            return

        self._is_cleanup = True

        try:
            # Clear figure instead of closing
            if hasattr(self, "figure") and self.figure:
                self.figure.clear()

            # Prevent canvas deletion
            if hasattr(self, "canvas") and self.canvas:
                try:
                    self.canvas.draw_idle()
                except Exception as draw_error:
                    print(f"Error in canvas draw_idle: {draw_error}")

        except Exception as e:
            print(f"Warning in cleanup: {e}")

    def __del__(self):
        """Safe cleanup on deletion"""
        try:
            if not self._is_cleanup:
                self.cleanup()
        except Exception as e:
            print(f"Warning in __del__: {e}")
