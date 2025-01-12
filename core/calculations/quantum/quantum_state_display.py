# quantum_state_display.py
# Quantum State Visualization Module
#
# This module handles the visualization of quantum states, including:
# - Phase distribution plots
# - State probability visualization
# - Coherence metrics display
# - Pythagorean influence visualization
#
# The visualizer manages both the creation and updating of plots using matplotlib.
import matplotlib.pyplot as plt
import numpy as np


class QuantumStateVisualizer:
    def __init__(self, figure):
        self.figure = figure

    def plot_quantum_state(self, results, circuit=None, pythagorean_results=None):
        """Enhanced visualization with grid layout"""
        self.figure.clear()
        self.figure.set_size_inches(15, 12)

        gs = self.figure.add_gridspec(
            2, 2, height_ratios=[3, 3], width_ratios=[3, 3], hspace=0.6, wspace=0.6
        )

        # 1. Phase Distribution (Top Left)
        ax_phase = self.figure.add_subplot(gs[0, 0], projection="polar")
        self._plot_phase_distribution(ax_phase, results)

        # 2. Quantum Surface (Top Right)
        ax_surface = self.figure.add_subplot(gs[0, 1], projection="3d")
        self._plot_quantum_surface(ax_surface)

        # 3. State Probabilities (Bottom Left)
        ax_prob = self.figure.add_subplot(gs[1, 0])
        self._plot_state_probabilities(ax_prob, results.get("counts", {}))

        # 4. Coherence Metrics (Bottom Right)
        ax_metrics = self.figure.add_subplot(gs[1, 1])
        self._plot_coherence_metrics(ax_metrics, results)

    def _plot_phase_distribution(self, ax, results):
        """Plot phase distribution in polar form"""
        try:
            phases = np.array(results.get("phases", [0]))
            magnitudes = np.array(results.get("statevector", [1]))

            # Ensure arrays are the same length
            min_len = min(len(phases), len(magnitudes))
            phases = phases[:min_len]
            magnitudes = magnitudes[:min_len]

            # Handle empty arrays
            if len(phases) == 0 or len(magnitudes) == 0:
                phases = np.array([0])
                magnitudes = np.array([0])

            scatter = ax.scatter(
                phases, magnitudes, c=magnitudes, cmap="viridis", s=100, alpha=0.7
            )

            ax.set_title("Phase Distribution")
            ax.grid(True, alpha=0.3)
            self.figure.colorbar(scatter, ax=ax, label="Magnitude")
        except Exception as e:
            print(f"Error in phase distribution plot: {e}")
            ax.text(
                0.5, 0.5, "Error plotting phase distribution", ha="center", va="center"
            )

    def _plot_quantum_surface(self, ax):
        try:
            x_coords = np.linspace(-5, 5, 50)
            y_coords = np.linspace(-5, 5, 50)
            grid_x, grid_y = np.meshgrid(x_coords, y_coords)
            grid_z = np.sin(np.sqrt(grid_x**2 + grid_y**2)) * np.exp(
                -0.1 * (grid_x**2 + grid_y**2)
            )

            surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap="viridis", alpha=0.7)
            ax.set_title("Quantum State Surface")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            self.figure.colorbar(surf, ax=ax, label="Amplitude")
        except Exception as e:
            print(f"Error in quantum surface plot: {e}")
            ax.text(0.5, 0.5, "Error plotting surface", ha="center", va="center")

    def _plot_state_probabilities(self, ax, counts):
        try:
            if not counts:
                ax.text(
                    0.5, 0.5, "No probability data available", ha="center", va="center"
                )
                ax.set_title("State Probabilities")
                return

            states = list(counts.keys())
            probabilities = [count / sum(counts.values()) for count in counts.values()]
            colors = plt.cm.viridis(np.linspace(0, 1, len(states)))

            bars = ax.bar(states, probabilities, color=colors)
            ax.set_title("State Probabilities")
            ax.set_xlabel("States")
            ax.set_ylabel("Probability")
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", rotation=45)
        except Exception as e:
            print(f"Error in probability plot: {e}")
            ax.text(0.5, 0.5, "Error plotting probabilities", ha="center", va="center")

    def _plot_coherence_metrics(self, ax, results):
        try:
            metrics = [
                'Quantum Purity\n(How "clean" the state is)',
                "Quantum Fidelity\n(How well preserved)",
            ]
            values = [results.get("purity", 0), results.get("fidelity", 1.0)]
            colors = plt.cm.viridis([0.2, 0.8])

            bars = ax.bar(metrics, values, color=colors)
            ax.set_title("Quantum Quality Metrics")
            ax.set_ylim(0, 1.1)

            # for bar in bars:
            #     height = bar.get_height()
            #     ax.text(bar.get_x() + bar.get_width() / 2., height,
            #             f'{height:.3f}\nGreat ↑\nPoor ↓',
            #             ha='center', va='bottom')
        except Exception as e:
            print(f"Error in metrics plot: {e}")
            ax.text(0.5, 0.5, "Error plotting metrics", ha="center", va="center")
