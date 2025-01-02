# quantum_harmonic_analysis.py
# Quantum Harmonics Analysis Module
# This module provides the core analysis functionality for quantum harmonic systems.
# It handles quantum circuit simulation, material analysis, and quantum state calculations.
#
# Key Components:
# - Quantum circuit simulation with noise modeling
# - Material property analysis and scoring
# - Harmonic frequency analysis with quantum states
#
# The analyzer integrates with the quantum materials API and provides comprehensive
# analysis results including state probabilities, phases, and material compatibility.

# Qiskit SDK imports
from qiskit import transpile
from qiskit_aer import AerSimulator
from data.elements import atomic_frequencies
from data.frequencies import frequency_systems
from core.calculations.quantum.pythagorean_analysis import PythagoreanAnalyzer
from core.calculations.quantum.quantum_circuit_builder import create_circuit
import numpy as np


class AtomicFrequencyAnalyzer:
    def __init__(self):
        self.atomic_data = atomic_frequencies
        self.frequency_systems = frequency_systems
        self.audible_range = (20, 20000)

    def analyze_frequency_spectrum(self, frequencies, amplitudes):
        """Analyze frequencies using atomic data and musical systems"""
        analysis = []
        for freq, amp in zip(frequencies, amplitudes):
            matches = self._find_atomic_matches(freq)
            if matches:
                analysis.append(
                    {
                        "frequency": freq,
                        "amplitude": amp,
                        "atomic_matches": matches,
                        "musical_mappings": self.map_frequency_to_notes(freq),
                    }
                )
        return analysis

    def map_frequency_to_notes(self, frequency):
        """Map frequency to notes in different musical systems"""
        mappings = {}

        for system_name, notes in self.frequency_systems.items():
            closest_note = None
            min_diff = float("inf")

            for note, note_freq in notes.items():
                diff = abs(frequency - note_freq)
                if diff < min_diff:
                    min_diff = diff
                    closest_note = note

            if closest_note:
                mappings[system_name] = {
                    "note": closest_note,
                    "frequency": notes[closest_note],
                    "deviation": min_diff,
                }

        return mappings

    def _find_atomic_matches(self, frequency):
        """Find atomic elements matching the frequency - internal helper method"""
        matches = []
        for element, mass in self.atomic_data.items():
            harmonic = self._find_matching_harmonic(frequency, mass)
            if harmonic:
                matches.append(
                    {
                        "element": element,
                        "harmonic": harmonic["n"],
                        "base_frequency": harmonic["base_freq"],
                    }
                )
        return matches

    def _find_matching_harmonic(self, frequency, mass):
        """Find matching harmonic - internal helper method"""
        mass_kg = mass * 1.660539067e-27
        base_freq = (mass_kg * 299792458**2) / 6.62607015e-34

        for n in range(1, 21):
            harmonic_freq = base_freq / n
            if abs(harmonic_freq - frequency) < 1.0:
                return {"n": n, "base_freq": base_freq}
        return None


class QuantumHarmonicsAnalyzer:
    def __init__(self):
        self.sim = AerSimulator()
        self.pythagorean_analyzer = PythagoreanAnalyzer()

    # In quantum_harmonic_analysis.py
    def analyze_harmonics(
        self, frequencies, amplitudes, internal_noise=0, external_noise=0
    ):
        """Main analysis combining quantum, atomic, and Pythagorean analysis"""
        try:
            # Create and run quantum circuit
            circuit = create_circuit(frequencies, amplitudes)[
                0
            ]  # Get just the circuit from tuple
            transpiled = transpile(circuit, self.sim)
            result = self.sim.run(transpiled, shots=1000).result()
            counts = result.get_counts()

            # Calculate quantum states and phases
            probabilities = {state: count / 1000 for state, count in counts.items()}
            phases = [np.angle(f + 1j * a) for f, a in zip(frequencies, amplitudes)]

            # Get Pythagorean analysis
            pythagorean_results = self.pythagorean_analyzer.analyze_harmonics(
                frequencies
            )

            # Analyze atomic frequencies
            atomic_analysis = self._analyze_atomic_frequencies(frequencies, amplitudes)

            # Combine results
            return {
                "circuit": circuit,
                "counts": counts,
                "probabilities": probabilities,
                "phases": phases,
                "statevector": np.array([np.sqrt(p) for p in probabilities.values()]),
                "purity": sum(p * p for p in probabilities.values()),
                "fidelity": 1.0 - (internal_noise + external_noise) / 2,
                "atomic_analysis": atomic_analysis,
                "pythagorean_analysis": pythagorean_results,
            }

        except Exception as e:
            print(f"Error in harmonics analysis: {str(e)}")
            raise

    def _analyze_atomic_frequencies(self, frequencies, amplitudes):
        """Analyze frequencies for atomic and musical relationships"""
        analysis_results = []

        for freq, amp in zip(frequencies, amplitudes):
            # Find atomic element matches
            atomic_matches = []
            for element, mass in atomic_frequencies.items():
                harmonic = self._find_matching_harmonic(freq, mass)
                if harmonic:
                    atomic_matches.append(
                        {
                            "element": element,
                            "harmonic": harmonic["n"],
                            "base_frequency": harmonic["base_freq"],
                        }
                    )

            # Find musical note mappings
            musical_mappings = {}
            for system_name, notes in frequency_systems.items():
                closest_note = None
                min_diff = float("inf")

                for note, note_freq in notes.items():
                    diff = abs(freq - note_freq)
                    if diff < min_diff:
                        min_diff = diff
                        closest_note = note

                if closest_note:
                    musical_mappings[system_name] = {
                        "note": closest_note,
                        "frequency": notes[closest_note],
                        "deviation": min_diff,
                    }

            if atomic_matches or musical_mappings:
                analysis_results.append(
                    {
                        "frequency": freq,
                        "amplitude": amp,
                        "atomic_matches": atomic_matches,
                        "musical_mappings": musical_mappings,
                    }
                )

        return analysis_results

    def _find_matching_harmonic(self, frequency, mass):
        """Find matching harmonic for given frequency and mass"""
        # Convert mass to base frequency using E = hf = mc²
        mass_kg = mass * 1.660539067e-27  # Convert atomic mass to kg
        base_freq = (mass_kg * 299792458**2) / 6.62607015e-34

        # Check first 20 harmonics
        for n in range(1, 21):
            harmonic_freq = base_freq / n
            if abs(harmonic_freq - frequency) < 1.0:  # 1 Hz tolerance
                return {"n": n, "base_freq": base_freq}
        return None
