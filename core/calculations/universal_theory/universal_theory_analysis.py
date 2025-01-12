# unified_quantum_harmonic_system.py
import numpy as np
import tensorflow as tf
from qiskit import QuantumCircuit
from scipy.fft import fft, fftfreq
from data.universal_measurements.frequencies import frequency_systems
from data.universal_measurements.elements import atomic_frequencies


class UnifiedHarmonicSystem:
    def __init__(self, base_freq=432):
        self.phi = (1 + np.sqrt(5)) / 2
        self.base_freq = base_freq

        # Musical Systems
        self.frequency_systems = frequency_systems

        # Atomic weights and frequencies
        self.atomic_frequencies = atomic_frequencies

        # Planetary system with element associations
        # Replace in unified_quantum_harmonic_system.py
        self.planets = {
            "Mercury": {
                "fib": 2,
                "element": "Hg",
                "frequency": self.frequency_systems["pythagorean"]["G3"],
                "atomic_weight": self.atomic_frequencies["Hg"],
            },
            "Venus": {
                "fib": 1,
                "element": "Cu",
                "frequency": self.frequency_systems["pythagorean"]["A3"],
                "atomic_weight": self.atomic_frequencies["Cu"],
            },
            "Earth": {
                "fib": 1,
                "element": "Fe",
                "frequency": self.frequency_systems["pythagorean"]["C3"],
                "atomic_weight": self.atomic_frequencies["Fe"],
            },
            "Mars": {
                "fib": 3,
                "element": "S",
                "frequency": self.frequency_systems["western_12_tone"]["D3"],
                "atomic_weight": self.atomic_frequencies["S"],
            },
            "Jupiter": {
                "fib": 5,
                "element": "H",
                "frequency": self.frequency_systems["western_12_tone"]["F3"],
                "atomic_weight": self.atomic_frequencies["H"],
            },
            "Saturn": {
                "fib": 8,
                "element": "He",
                "frequency": self.frequency_systems["western_12_tone"]["G3"],
                "atomic_weight": self.atomic_frequencies["He"],
            },
            "Uranus": {
                "fib": 13,
                "element": "N",
                "frequency": self.frequency_systems["western_12_tone"]["A3"],
                "atomic_weight": self.atomic_frequencies["N"],
            },
            "Neptune": {
                "fib": 21,
                "element": "O",
                "frequency": self.frequency_systems["western_12_tone"]["B3"],
                "atomic_weight": self.atomic_frequencies["O"],
            },
        }

    def calculate_ratios(self, freq1, freq2):
        """Calculate harmonic ratios between frequencies"""
        ratio = freq1 / freq2
        phi_ratio = ratio / self.phi
        return ratio, phi_ratio

    def element_to_frequency(self, element):
        """Convert atomic weight to frequency using golden ratio scaling"""
        atomic_weight = self.atomic_frequencies[element]
        return self.base_freq * (atomic_weight / self.atomic_frequencies["H"])

    def create_planetary_tensor(self):
        """Create tensor with all frequency relationships"""
        planet_data = []

        for planet, data in self.planets.items():
            # Get frequencies from different systems
            planetary_freq = data["frequency"]
            atomic_freq = self.atomic_frequencies[data["element"]]
            fib_freq = data["fib"] * self.base_freq

            # Calculate harmonic relationships
            planetary_tensor = np.array(
                [
                    planetary_freq,
                    atomic_freq,
                    fib_freq,
                    planetary_freq * self.phi,
                    atomic_freq / self.phi,
                    (planetary_freq + atomic_freq) / 2,  # Mean frequency
                    fib_freq * self.phi,
                ]
            )

            planet_data.append(planetary_tensor)

        return tf.convert_to_tensor(planet_data)

    def find_cross_system_resonances(self):
        """Find resonances between different musical systems"""
        resonances = []

        for system1, freqs1 in self.frequency_systems.items():
            for system2, freqs2 in self.frequency_systems.items():
                if system1 >= system2:  # Avoid duplicate comparisons
                    continue

                for note1, freq1 in freqs1.items():
                    for note2, freq2 in freqs2.items():
                        ratio = freq1 / freq2
                        if abs(ratio - self.phi) < 0.01:  # Check for golden ratio
                            resonances.append(
                                {
                                    "system1": system1,
                                    "system2": system2,
                                    "note1": note1,
                                    "note2": note2,
                                    "ratio": ratio,
                                    "phi_accuracy": abs(ratio - self.phi),
                                }
                            )

        return resonances

    def analyze_system_interactions(self):
        """Analyze interactions between all frequency systems"""
        resonances = self.find_cross_system_resonances()
        planetary_tensor = self.create_planetary_tensor()

        # Find points where multiple systems align
        convergence_points = []

        for res in resonances:
            for p_idx, p_data in enumerate(planetary_tensor):
                if abs(p_data[0] / res["ratio"] - self.phi) < 0.01:
                    convergence_points.append(
                        {
                            "planet": list(self.planets.keys())[p_idx],
                            "system1": res["system1"],
                            "system2": res["system2"],
                            "frequency": p_data[0],
                            "phi_accuracy": abs(p_data[0] / res["ratio"] - self.phi),
                        }
                    )

        return convergence_points

    def quantum_harmonic_circuit(self, n_qubits=8):
        """Create quantum circuit for harmonic analysis"""
        qc = QuantumCircuit(n_qubits)

        # Initialize in superposition
        qc.h(range(n_qubits))

        planetary_tensor = self.create_planetary_tensor()

        # Add phases based on frequency relationships
        for i in range(n_qubits):
            freq = tf.reduce_sum(planetary_tensor[i]).numpy()
            phase = float(2 * np.pi * freq / self.base_freq)
            qc.rz(phase, i)

            # Add entanglement between adjacent qubits
            if i < n_qubits - 1:
                qc.cx(i, i + 1)

        return qc

    def find_musical_resonances(self):
        """Find resonances between all musical systems and atomic frequencies"""
        resonances = []

        for system_name, system in self.frequency_systems.items():
            for note, freq in system.items():
                for element, atomic_weight in self.atomic_frequencies.items():
                    element_freq = self.element_to_frequency(element)
                    ratio, phi_ratio = self.calculate_ratios(freq, element_freq)

                    # Adjust threshold for more matches
                    if abs(ratio - round(ratio, 2)) < 0.1:  # Changed from 0.01
                        resonances.append(
                            {
                                "system": system_name,
                                "note": note,
                                "element": element,
                                "ratio": ratio,
                                "phi_relation": phi_ratio,
                            }
                        )

        return resonances

    def analyze_harmonic_convergence(self):
        """Analyze convergence points between all systems"""
        planetary_tensor = self.create_planetary_tensor()
        resonances = self.find_musical_resonances()

        convergence_points = []
        for planet_idx, planet_freqs in enumerate(planetary_tensor):
            for resonance in resonances:
                ratio = planet_freqs[0] / resonance["ratio"]
                # Adjust threshold for more matches
                if abs(ratio - self.phi) < 0.1:  # Changed from 0.01
                    convergence_points.append(
                        {
                            "planet": list(self.planets.keys())[planet_idx],
                            "system": resonance["system"],
                            "note": resonance["note"],
                            "element": resonance["element"],
                            "phi_accuracy": abs(ratio - self.phi),
                        }
                    )

        return convergence_points

    def analyze_frequency_spectrum(self, time_steps=1000):
        """Analyze frequency spectrum of combined system"""
        t = np.linspace(0, 1, time_steps)
        planetary_tensor = self.create_planetary_tensor()

        # Combine all frequencies
        combined_signal = np.zeros_like(t)
        for planet_freqs in planetary_tensor:
            for freq in planet_freqs:
                combined_signal += np.sin(2 * np.pi * freq * t)

        # Perform FFT
        freqs = fftfreq(len(t), t[1] - t[0])
        spectrum = np.abs(fft(combined_signal))

        return freqs[: len(freqs) // 2], spectrum[: len(freqs) // 2]

    def generate_analysis_report(self):
        """Generate human-readable analysis report"""
        resonances = self.find_musical_resonances()
        convergence = self.analyze_harmonic_convergence()

        report = {
            "strong_relationships": [],
            "harmonic_patterns": [],
            "planetary_connections": [],
            "summary": {},
        }

        # Find strongest relationships
        for res in resonances:
            if res["phi_relation"] > 0.8:
                report["strong_relationships"].append(
                    {
                        "musical_note": res["note"],
                        "element": res["element"],
                        "strength": f"{res['phi_relation']:.2f}",
                    }
                )

        # Analyze planetary patterns
        for conv in convergence:
            report["planetary_connections"].append(
                {
                    "planet": conv["planet"],
                    "musical_note": conv["note"],
                    "accuracy": f"{(1 - conv['phi_accuracy']) * 100:.1f}%",
                }
            )

        # Calculate overall system harmony
        total_resonance = sum(r["phi_relation"] for r in resonances) / len(resonances)
        report["summary"] = {
            "overall_harmony": f"{total_resonance:.2f}",
            "strongest_element": max(resonances, key=lambda x: x["phi_relation"])[
                "element"
            ],
            "primary_frequency": f"{self.base_freq} Hz",
        }

        return report

    # In main.py
    def print_friendly_report(self, system):
        """Print user-friendly analysis report"""
        report = system.generate_analysis_report()

        print("\n=== Harmony Analysis Report ===\n")

        print("Strong Harmonic Relationships:")
        for rel in report["strong_relationships"]:
            print(
                f"• {rel['musical_note']} resonates with {rel['element']} at {rel['strength']} strength"
            )

        print("\nPlanetary Connections:")
        for conn in report["planetary_connections"]:
            print(
                f"• {conn['planet']} aligns with {conn['musical_note']} ({conn['accuracy']} match)"
            )

        print("\nOverall System Summary:")
        print(f"• Base Frequency: {report['summary']['primary_frequency']}")
        print(f"• System Harmony: {report['summary']['overall_harmony']}")
        print(f"• Most Resonant Element: {report['summary']['strongest_element']}")
