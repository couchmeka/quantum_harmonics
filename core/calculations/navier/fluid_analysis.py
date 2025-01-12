# fluid_analyzer.py, defines the FluidAnalyzer class that integrates
# concepts from fluid dynamics,quantum mechanics, and music theory to
# analyze complex fluid systems influenced by quantum effects. It provides tools to
# model Navier-Stokes equations incorporating quantum potentials, and performs sophisticated
# analyses using physical constants and computed parameters like decoherence times and
# phonon coupling. Additionally, it extends its application to acoustic physics by
# examining harmonic relationships in music through Fibonacci sequences and Pythagorean tuning.
# This class is aimed at researchers and engineers needing to simulate
# and analyze fluid systems under quantum influences
# and harmonic principles.

import numpy as np
from PyQt6.QtCore import pyqtSignal
from scipy.integrate import odeint


class FluidAnalyzer:
    def __init__(self, frequency_systems, atomic_frequencies):
        self.frequency_systems = frequency_systems
        self.atomic_frequencies = atomic_frequencies
        self.density = 1.225
        self.viscosity = 1.81e-5
        self.speed_of_sound = 343
        self.hbar = 1.0545718e-34
        self.decoherence_time = 1e-12
        self.material_mass = 1.6605e-27  # mass (kg, default to proton mass)
        self.phonon_coupling = 1e-12  # phonon coupling strength
        self.lattice_constant = 5e-10  # typical lattice spacing (m)

        self.decoherence_time = self.hbar / (
            self.phonon_coupling * self.lattice_constant**2
        )

    def quantum_tunneling_effect(self, fluid_state, barrier_height):
        # Prevent overflow in exponential
        exponent = np.clip(
            -np.sqrt(2 * barrier_height) * fluid_state / self.hbar, -700, 700
        )
        psi = np.exp(exponent)
        # Prevent overflow in square operation
        return np.minimum(np.abs(psi) ** 2, 1.0)  # Clip to maximum probability of 1

    def calculate_quantum_effects(self, fluid_state, time):
        # Add temperature parameter (in Kelvin)
        T = 300  # room temperature
        kb = 1.380649e-23  # Boltzmann constant

        # Thermal decoherence time
        thermal_decoherence = self.hbar / (kb * T)

        # Calculate coherence with thermal effects
        omega = 2 * np.pi * np.mean(fluid_state)
        coherence = np.cos(omega * time) * np.exp(-time / thermal_decoherence)

        return np.abs(coherence) * np.max(fluid_state)

    def navier_stokes_advanced(self, y, t, params, frequencies=None):
        vx, vy, vz, p = y[0:3], y[3:6], y[6:9], y[9:12]
        L, Re, Ma = params

        # Add stability constraints
        max_value = 0.001
        vx = np.clip(vx, -max_value, max_value)
        vy = np.clip(vy, -max_value, max_value)
        vz = np.clip(vz, -max_value, max_value)

        # Add this pressure coupling
        pressure_coupling = (
            Ma
            * self.speed_of_sound
            * (
                np.sum(np.gradient(vx))
                + np.sum(np.gradient(vy))
                + np.sum(np.gradient(vz))
            )
        )

        # Modify pressure calculation
        dp = (
            -self.speed_of_sound**2
            * (np.gradient(vx) + np.gradient(vy) + np.gradient(vz))
            + pressure_coupling
        )

        # Single quantum potential calculation
        quantum_potential = (
            -self.hbar**2
            * Ma
            / (2 * self.density)
            * np.gradient(np.gradient(p + 1e-10))
        )

        # Ensure pressure remains positive
        p = np.maximum(p + quantum_potential + pressure_coupling, 1e-10)

        # Scale convective terms by Reynolds number
        dvx = -(1 / Re) * (
            vx * np.gradient(vx) + vy * np.gradient(vy) + vz * np.gradient(vz)
        )
        dp = -np.gradient(p)

        # Scale viscous terms by length and Reynolds number
        laplacian_v = (1 / (L**2 * Re)) * (
            np.gradient(np.gradient(vx))
            + np.gradient(np.gradient(vy))
            + np.gradient(np.gradient(vz))
        )

        if frequencies is not None:
            frequency_term = np.sum([np.sin(2 * np.pi * f * t) for f in frequencies])
            dvx += frequency_term * Ma
            p += frequency_term * self.density * self.speed_of_sound * Ma

        dvx += -(1 / self.density) * dp[0] + laplacian_v
        dvy = -(1 / self.density) * dp[1] + laplacian_v
        dvz = -(1 / self.density) * dp[2] + laplacian_v

        dp = (
            -Ma
            * self.speed_of_sound**2
            * (np.gradient(vx) + np.gradient(vy) + np.gradient(vz))
        )

        # Clip final values
        dvx = np.clip(dvx, -max_value, max_value)
        dvy = np.clip(dvy, -max_value, max_value)
        dvz = np.clip(dvz, -max_value, max_value)
        dp = np.clip(dp, -max_value, max_value)

        return np.concatenate([dvx, dvy, dvz, dp])

    # First, make fibonacci_quantum_ratio a static method since it doesn't use self
    @staticmethod
    def fibonacci_quantum_ratio(n: int):
        # Generate Fibonacci sequence
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i - 1] + fib[i - 2])

        # Calculate quantum frequency ratios
        ratios = [fib[i + 1] / fib[i] for i in range(len(fib) - 1)]
        return ratios

    def analyze_fluid_dynamics(
        self, notes=None, Re=1000, Ma=0.3, pythagorean_ratio=1.5
    ):
        t = np.linspace(0, 1, 1000)

        # Add initial perturbation to pressure
        initial_pressure = (
            self.density * self.speed_of_sound**2 * (1 + 0.1 * np.random.rand(3))
        )

        y0 = np.concatenate([np.zeros(3), np.zeros(3), np.zeros(3), initial_pressure])

        frequencies = []
        pythagorean_frequencies = []

        # Generate Fibonacci ratios - Fix: Call the method with parameter
        fibonacci_ratios = self.fibonacci_quantum_ratio(8)  # Changed this line
        fibonacci_frequencies = []

        if notes:
            for note in notes:
                note = note.strip()
                for system_name, system in self.frequency_systems.items():
                    if note in system:
                        freq = system[note]
                        frequencies.append(freq)
                        # Add Pythagorean harmonic
                        pythagorean_frequencies.append(freq * pythagorean_ratio)
                        # Add Fibonacci harmonics
                        fibonacci_frequencies.extend(
                            [freq * ratio for ratio in fibonacci_ratios]
                        )
                        break

        # Combine all frequencies
        all_frequencies = frequencies + pythagorean_frequencies + fibonacci_frequencies

        params = (1.0, Re, Ma)
        solution = odeint(
            self.navier_stokes_advanced, y0, t, args=(params, all_frequencies)
        )

        return {
            "t": t,
            "solution": solution,
            "original_frequencies": frequencies,
            "pythagorean_frequencies": pythagorean_frequencies,
            "fibonacci_frequencies": fibonacci_frequencies,
            "system_matches": self.find_system_matches(all_frequencies),
        }

    def find_system_matches(self, frequencies):
        matches = {}
        for freq in frequencies:
            for system_name, system in self.frequency_systems.items():
                for note, note_freq in system.items():
                    if abs(freq - note_freq) < 0.1:
                        if system_name not in matches:
                            matches[system_name] = []
                        matches[system_name].append(note)
        return matches

    def plot_pythagorean_resonance(self, frequencies, ratio):
        resonance = []
        for f1 in frequencies:
            for f2 in frequencies:
                if f1 < f2:
                    actual_ratio = f2 / f1
                    resonance.append(abs(actual_ratio - ratio))
        return np.mean(resonance)  # Return average resonance deviation

    def run_analysis(self):
        try:
            results = self.analyzer.analyze_fluid_dynamics(
                notes=self.note_input.text().split(","),
                Re=self.re_slider.value(),
                Ma=self.ma_slider.value() / 100.0,
                pythagorean_ratio=self.ratio_slider.value() / 10.0,
            )
            self.current_results = {
                "original_frequencies": results["original_frequencies"],
                "fibonacci_frequencies": results["fibonacci_frequencies"],
                "solution": results["solution"],
                "t": results["t"],
            }
            self.analysis_complete.emit(self.current_results)
            self.plot_results(results)
        except Exception as e:
            print(f"Analysis error: {str(e)}")
