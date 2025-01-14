import numpy as np
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from data.backend_data_management.data_manager import QuantumDataManager


class QuantumLagrangianAnalyzer:
    def __init__(self, data_manager):
        data_manager = QuantumDataManager()  # Your existing data manager
        self.data_manager = data_manager
        self.figure, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        plt.tight_layout()

    def calculate_lagrangian(self, positions, velocities, mass=1.0, k=1.0):
        """
        Calculate the Lagrangian for harmonic motion.
        :param positions: Array of positions of particles
        :param velocities: Array of velocities of particles
        :param mass: Mass of the particles
        :param k: Spring constant (harmonic potential)
        :return: Lagrangian (T - V), T, V
        """
        kinetic_energy = 0.5 * mass * np.sum(velocities**2)
        potential_energy = 0.5 * k * np.sum(positions**2)
        lagrangian = kinetic_energy - potential_energy
        return lagrangian, kinetic_energy, potential_energy

    def analyze_quantum_state(self, state):
        """
        Analyze a quantum state.
        :param state: Quantum state (Statevector)
        :return: density_matrix, probabilities, entropy
        """
        density_matrix = np.outer(state.data, np.conj(state.data))
        probabilities = np.abs(state.data) ** 2
        entropy = -np.sum(
            probabilities * np.log2(probabilities + 1e-12)
        )  # Von Neumann entropy
        return density_matrix, probabilities, entropy

    def visualize(self, state, positions, velocities, timestep, T, V, L):
        """
        Visualize analysis results.
        :param state: Quantum state (Statevector)
        :param positions: Particle positions
        :param velocities: Particle velocities
        :param timestep: Current timestep
        :param T: Kinetic energy
        :param V: Potential energy
        :param L: Lagrangian
        """
        for ax in self.axes.ravel():
            ax.clear()

        # 1. Statevector Probabilities
        _, probabilities, entropy = self.analyze_quantum_state(state)
        self.axes[0, 0].bar(range(len(probabilities)), probabilities)
        self.axes[0, 0].set_title(f"Statevector Probabilities (Entropy: {entropy:.2f})")
        self.axes[0, 0].set_xlabel("State Index")
        self.axes[0, 0].set_ylabel("Probability")

        # 2. Phase Space
        self.axes[0, 1].plot(positions, velocities, "o-", label="Phase Space")
        self.axes[0, 1].set_title("Phase Space (Position vs Velocity)")
        self.axes[0, 1].set_xlabel("Position")
        self.axes[0, 1].set_ylabel("Velocity")

        # 3. Energy Landscape
        self.axes[1, 0].bar(
            ["Kinetic (T)", "Potential (V)", "Lagrangian (L)"], [T, V, L]
        )
        self.axes[1, 0].set_title("Energy Analysis")

        # 4. Density Matrix Heatmap
        density_matrix, _, _ = self.analyze_quantum_state(state)
        im = self.axes[1, 1].imshow(
            np.abs(density_matrix), cmap="viridis", interpolation="nearest"
        )
        self.axes[1, 1].set_title("Density Matrix Heatmap")
        self.figure.colorbar(im, ax=self.axes[1, 1])

        plt.suptitle(f"Timestep {timestep}")
        plt.draw()
        plt.pause(0.1)

    def simulate_and_analyze(self, timesteps=10, dt=0.1, mass=1.0, k=1.0):
        """
        Simulate harmonic motion and analyze quantum states.
        :param timesteps: Number of timesteps
        :param dt: Time step size
        :param mass: Mass of the particles
        :param k: Spring constant
        """
        positions = np.random.rand(3)  # Initial positions
        velocities = np.random.rand(3)  # Initial velocities

        for t in range(timesteps):
            # Calculate Lagrangian and energies
            L, T, V = self.calculate_lagrangian(positions, velocities, mass, k)
            print(f"Timestep {t}: L = {L:.2f}, T = {T:.2f}, V = {V:.2f}")

            # Update positions and velocities
            accelerations = -k * positions / mass
            velocities += accelerations * dt
            positions += velocities * dt

            # Get latest quantum state
            quantum_data = self.data_manager.get_latest_quantum_results()
            if quantum_data and "statevector" in quantum_data:
                state = Statevector(quantum_data["statevector"])
                self.visualize(state, positions, velocities, t, T, V, L)
