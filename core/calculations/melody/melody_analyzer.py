# This Python file, melody_analyzer.py, defines the MelodyAnalyzer class,
# which integrates physics and music theory to analyze and visualize the
# harmonies of musical notes through quantum mechanics and fluid dynamics models.
# It calculates Hamiltonian matrices, solves Navier-Stokes equations for sound wave propagation,
# and provides comprehensive visualizations of frequencies, eigenstates,
# wave evolutions, and phase space trajectories. The class utilizes scientific libraries
# such as NumPy, Matplotlib, and SciPy to perform these analyses and is
# designed for applications in music theory, acoustics research,
# and educational purposes in physics and engineering.

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import odeint
from data.universal_measurements.frequencies import frequency_systems
from data.universal_measurements.elements import atomic_frequencies


class MelodyAnalyzer:
    def __init__(self):
        self.frequency_systems = frequency_systems
        self.atomic_frequencies = atomic_frequencies

        # Physical constants
        self.hbar = 1.0545718e-34  # Reduced Planck constant
        self.viscosity = 1.81e-5  # Air viscosity
        self.density = 1.225  # Air density
        self.speed_of_sound = 343  # m/s

    def create_hamiltonian(self, frequencies):
        """
        Create Hamiltonian matrix for the system
        H = T + V where T is kinetic and V is potential energy
        """
        N = len(frequencies)
        H = np.zeros((N, N), dtype=complex)

        # Kinetic energy terms (diagonal)
        for i in range(N):
            H[i, i] = frequencies[i] * self.hbar

        # Potential energy terms (off-diagonal coupling)
        for i in range(N - 1):
            coupling = np.sqrt(frequencies[i] * frequencies[i + 1]) * self.hbar / 2
            H[i, i + 1] = coupling
            H[i + 1, i] = coupling.conjugate()

        return H

    def navier_stokes_sound(self, y, t, L):
        """
        Simplified Navier-Stokes equations for sound wave propagation
        y = [velocity, pressure]
        """
        velocity, pressure = y

        # Continuity equation
        dvelocity = -1 / (self.density * L) * pressure

        # Momentum equation with viscosity
        dpressure = (
            -self.density * self.speed_of_sound**2 / L * velocity
            + self.viscosity / L**2 * velocity
        )

        return [dvelocity, dpressure]

    def analyze_harmonies(self, input_notes, system="western_12_tone"):
        """Analyze input notes with quantum and fluid dynamics"""
        analysis = {
            "notes": input_notes,
            "frequencies": [],
            "eigenstates": [],
            "wave_evolution": [],
            "phase_space": [],
        }

        # Get frequencies
        frequencies = [self.frequency_systems[system][note] for note in input_notes]
        analysis["frequencies"] = frequencies

        # Quantum analysis
        hamiltonian = self.create_hamiltonian(frequencies)
        eigenvalues, eigenvectors = eigh(hamiltonian)
        analysis["eigenstates"] = {"energies": eigenvalues, "states": eigenvectors}

        # Sound wave evolution
        t = np.linspace(0, 1, 1000)
        L = self.speed_of_sound / min(frequencies)  # Characteristic length
        y0 = [0, max(frequencies)]  # Initial conditions

        wave_solution = odeint(self.navier_stokes_sound, y0, t, args=(L,))
        analysis["wave_evolution"] = {
            "time": t,
            "velocity": wave_solution[:, 0],
            "pressure": wave_solution[:, 1],
        }

        # Phase space trajectory
        phase_space = np.zeros((len(frequencies), 2))
        for i, freq in enumerate(frequencies):
            phase_space[i] = [np.real(eigenvectors[i, 0]), np.imag(eigenvectors[i, 0])]
        analysis["phase_space"] = phase_space

        return analysis

    def visualize_analysis(self, analysis, figure=None):
        """Create four key visualizations"""
        if figure is None:
            figure = plt.figure(figsize=(15, 10))

        # 1. Frequency-Element Resonance
        ax1 = figure.add_subplot(221)
        frequencies = analysis["frequencies"]
        atomic_weights = list(self.atomic_frequencies.values())
        resonance_matrix = np.zeros((len(frequencies), len(atomic_weights)))

        for i, freq in enumerate(frequencies):
            for j, weight in enumerate(atomic_weights):
                resonance = abs(freq - weight * 440) / (freq + weight * 440)
                resonance_matrix[i, j] = resonance

        im1 = ax1.imshow(resonance_matrix, aspect="auto", cmap="viridis")
        ax1.set_xlabel("Elements")
        ax1.set_ylabel("Input Notes")
        ax1.set_title("Frequency-Element Resonance")
        figure.colorbar(im1, ax=ax1)

        # 2. Energy Level Diagram
        ax2 = figure.add_subplot(222)
        energies = analysis["eigenstates"]["energies"]
        for i, E in enumerate(energies):
            ax2.plot([-0.5, 0.5], [E, E], "b-")
        ax2.set_title("Quantum Energy Levels")
        ax2.set_ylabel("Energy (J)")
        ax2.set_xticks([])

        # 3. Wave Evolution
        ax3 = figure.add_subplot(223)
        t = analysis["wave_evolution"]["time"]
        v = analysis["wave_evolution"]["velocity"]
        p = analysis["wave_evolution"]["pressure"]
        ax3.plot(t, v, "b-", label="Velocity")
        ax3.plot(t, p, "r-", label="Pressure")
        ax3.set_title("Sound Wave Evolution")
        ax3.set_xlabel("Time (s)")
        ax3.legend()

        # 4. Phase Space Trajectory
        ax4 = figure.add_subplot(224)
        phase_space = analysis["phase_space"]
        ax4.plot(phase_space[:, 0], phase_space[:, 1], "bo-")
        ax4.set_title("Phase Space Trajectory")
        ax4.set_xlabel("Re(ψ)")
        ax4.set_ylabel("Im(ψ)")
        ax4.grid(True)

        figure.tight_layout()
        return figure
