import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from matplotlib.figure import Figure
from PyQt6.QtCore import QMutexLocker
import matplotlib.pyplot as plt
from core.calculations.particle.bloch_sphere_visualizer import BlochSphereVisualizer


class HigherDimensionalVisualizer(BlochSphereVisualizer):
    def __init__(self, data_manager, figure, canvas):
        super().__init__(data_manager, figure, canvas)
        self.n_qubits = 1  # Default to single qubit
        self.current_state = None
        self.layout_style = "horizontal"  # or 'grid'

    def set_num_qubits(self, n_qubits):
        """Set the number of qubits to visualize"""
        self.n_qubits = max(1, min(n_qubits, 4))  # Limit to 4 qubits for visualization
        self.setup_figure_layout()

    def setup_figure_layout(self):
        """Setup the figure layout based on number of qubits"""
        if self.layout_style == "grid":
            # Calculate grid dimensions
            grid_size = int(np.ceil(np.sqrt(self.n_qubits)))
            self.rows = grid_size
            self.cols = grid_size
        else:
            # Horizontal layout
            self.rows = 1
            self.cols = self.n_qubits

    def generate_higher_dimensional_state(self):
        """Generate a random quantum state for n qubits with physical constraints"""
        try:
            dim = 2**self.n_qubits

            # Generate random angles for spherical coordinates
            thetas = np.random.uniform(0, np.pi, dim)
            phis = np.random.uniform(0, 2 * np.pi, dim)

            # Convert to complex amplitudes
            state = np.zeros(dim, dtype=complex)
            for i in range(dim):
                state[i] = np.sin(thetas[i]) * np.exp(1j * phis[i])

            # Normalize
            state /= np.linalg.norm(state)

            return Statevector(state)

        except Exception as e:
            print(f"Error generating state: {e}")
            return None

    def analyze_state(self, state):
        """Analyze properties of the quantum state"""
        try:
            # Get the statevector data
            sv_data = state.data

            # Calculate various properties
            analysis = {
                "purity": float(state.purity()),
                "num_qubits": self.n_qubits,
                "entanglement": [],  # List to store pairwise entanglement
                "amplitudes": np.abs(sv_data) ** 2,  # Probability amplitudes
            }

            # Calculate pairwise entanglement if more than one qubit
            if self.n_qubits > 1:
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        # Could add entanglement measure here
                        analysis["entanglement"].append((i, j, 0.0))

            return analysis

        except Exception as e:
            print(f"Error analyzing state: {e}")
            return None

    def visualize_higher_dimensional_state(self, timestep=0):
        """Visualize a higher-dimensional quantum state with enhanced features"""
        print(f"Visualizing {self.n_qubits}-qubit system at timestep {timestep}...")

        try:
            with QMutexLocker(self._canvas_mutex):
                # Generate or update state
                self.current_state = self.generate_higher_dimensional_state()
                if self.current_state is None:
                    return

                # Analyze the state
                analysis = self.analyze_state(self.current_state)

                # Clear and setup figure
                self.figure.clear()

                # Create subplots based on layout
                for i in range(self.n_qubits):
                    ax = self.figure.add_subplot(
                        self.rows, self.cols, i + 1, projection="3d"
                    )

                    # Plot individual qubit state
                    plot_bloch_multivector(
                        self.current_state, title=f"Qubit {i+1}\nt={timestep}", ax=ax
                    )

                    # Add annotations
                    if analysis:
                        prob = analysis["amplitudes"][i]
                        ax.text2D(
                            0.05,
                            0.95,
                            f"P(|0⟩)={prob:.2f}",
                            transform=ax.transAxes,
                            color="white",
                        )

                # Add global state information
                if analysis:
                    info_text = f"System Purity: {analysis['purity']:.3f}\n"
                    if analysis["entanglement"]:
                        info_text += "Entanglement Present\n"
                    self.figure.text(
                        0.02,
                        0.02,
                        info_text,
                        color="white",
                        bbox=dict(facecolor="black", alpha=0.5),
                    )

                # Update display
                self.figure.tight_layout()
                if self.canvas:
                    self.canvas.draw()
                    self.canvas.flush_events()

                print(f"Visualization complete. Purity: {analysis['purity']:.3f}")

        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            traceback.print_exc()

    def update_animation(self, frame):
        """Override parent's animation update"""
        if frame is None:
            frame = 0
        self.animation_frame = frame
        self.visualize_higher_dimensional_state(timestep=frame)
        return []
