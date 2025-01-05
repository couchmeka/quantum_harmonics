from data.elements import atomic_frequencies


class AtomicResonanceAnalyzer:
    def __init__(self):
        self.h = 6.626e-34  # Planck's constant
        self.k = 1.380649e-23  # Boltzmann constant

    def analyze_atomic_resonances(self, frequencies, temperature=300, pressure=101325):
        """Analyzes frequencies for atomic resonances and harmonic relationships"""
        resonance_data = []

        for freq in frequencies:
            # Get element matches for this frequency
            element_matches = self.map_frequency_to_elements(freq)

            # Calculate basic quantum properties
            wavelength = 3e8 / freq  # c = fλ
            energy = freq * self.h  # E = hf

            # Calculate thermal effects
            thermal_energy = self.k * temperature
            thermal_broadening = 0.5 * freq * (thermal_energy / energy) ** 0.5

            # Pressure broadening (simplified model)
            pressure_broadening = freq * (pressure / 101325) * 1e-6

            resonance_data.append(
                {
                    "frequency": freq,
                    "wavelength": wavelength,
                    "energy": energy,
                    "elements": element_matches,
                    "environmental_effects": {
                        "temperature": temperature,
                        "thermal_broadening": thermal_broadening,
                        "pressure": pressure,
                        "pressure_broadening": pressure_broadening,
                    },
                    "strength": len(
                        element_matches
                    ),  # Simple measure of resonance strength
                }
            )

        # Calculate relationships between frequencies
        for i in range(len(frequencies)):
            for j in range(i + 1, len(frequencies)):
                ratio = min(frequencies[i], frequencies[j]) / max(
                    frequencies[i], frequencies[j]
                )

                # Find elements that resonate with both frequencies
                elements_i = set(
                    match["element"] for match in resonance_data[i]["elements"]
                )
                elements_j = set(
                    match["element"] for match in resonance_data[j]["elements"]
                )
                common_elements = elements_i.intersection(elements_j)

                if common_elements:
                    resonance_data[i].setdefault("harmonic_relationships", []).append(
                        {
                            "paired_frequency": frequencies[j],
                            "ratio": ratio,
                            "common_elements": list(common_elements),
                        }
                    )
                    resonance_data[j].setdefault("harmonic_relationships", []).append(
                        {
                            "paired_frequency": frequencies[i],
                            "ratio": ratio,
                            "common_elements": list(common_elements),
                        }
                    )

        return resonance_data

    def map_frequency_to_elements(self, frequency):
        """Maps a frequency to possible atomic elements using simplified resonance model"""
        matches = []
        target_energy = frequency * self.h  # E = hf

        for element, atomic_mass in atomic_frequencies.items():
            # Check harmonics 1-4
            for n1 in range(1, 6):
                for n2 in range(n1 + 1, 11):
                    transition_energy = self.calculate_transition_energy(
                        n1, n2, atomic_mass
                    )
                    if self.is_energy_match(target_energy, transition_energy):
                        matches.append(
                            {
                                "element": element,
                                "transition": f"{n1}→{n2}",
                                "energy_level": transition_energy,
                                "deviation": abs(target_energy - transition_energy)
                                / transition_energy,
                            }
                        )

        return matches

    def calculate_transition_energy(self, n1, n2, atomic_mass):
        """Calculate transition energy using atomic mass and harmonic transitions"""
        # Use atomic mass directly for energy calculation
        base_energy = atomic_mass * self.h  # E = mc²
        # Calculate transition energy between levels
        energy = base_energy * (1 / n1**2 - 1 / n2**2)
        return abs(energy)

    def is_energy_match(self, energy1, energy2, tolerance=0.1):
        """Check if two energies match within tolerance"""
        return abs(energy1 - energy2) / energy2 < tolerance
