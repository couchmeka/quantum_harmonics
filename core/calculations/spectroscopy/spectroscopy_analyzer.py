import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGroupBox
from data.elements import atomic_frequencies


class SpectroscopyAnalyzer:
    def __init__(self):
        self.planck = 6.626e-34  # Planck constant
        self.c = 3e8  # Speed of light
        self.k = 1.380649e-23  # Boltzmann constant

    def analyze_spectrum(self, frequencies, amplitudes, temperature=300):
        """Analyzes spectral data and atomic transitions"""
        spectral_data = []

        for freq, amp in zip(frequencies, amplitudes):
            # Calculate wavelength and energy
            wavelength = self.c / freq
            energy = freq * self.planck

            # Find atomic transitions
            transitions = self._find_atomic_transitions(energy)

            # Calculate thermal broadening
            thermal_width = self._calculate_thermal_width(freq, temperature)

            # Generate spectral line shape (Lorentzian profile)
            line_profile = self._generate_line_profile(freq, amp, thermal_width)

            spectral_data.append(
                {
                    "frequency": freq,
                    "wavelength": wavelength,
                    "energy": energy,
                    "amplitude": amp,
                    "transitions": transitions,
                    "thermal_width": thermal_width,
                    "line_profile": line_profile,
                }
            )

        return spectral_data

    def _find_atomic_transitions(self, energy):
        """Identify possible atomic transitions"""
        matches = []

        for element, mass in atomic_frequencies.items():
            # Convert mass to energy using E=mc²
            base_energy = mass * self.planck * self.c * self.c

            # Check various transition levels
            for n1 in range(1, 4):
                for n2 in range(n1 + 1, 5):
                    transition_energy = base_energy * (1 / n1**2 - 1 / n2**2)
                    if abs(transition_energy - energy) / energy < 0.1:  # 10% tolerance
                        matches.append(
                            {
                                "element": element,
                                "transition": f"{n1}→{n2}",
                                "energy_level": transition_energy,
                            }
                        )

        return matches

    def _calculate_thermal_width(self, frequency, temperature):
        """Calculate thermal broadening of spectral lines"""
        return frequency * np.sqrt(
            2 * self.k * temperature / (self.planck * self.c * self.c)
        )

    def _generate_line_profile(self, center_freq, amplitude, width, num_points=1000):
        """Generate Lorentzian line profile"""
        freq_range = np.linspace(
            center_freq - 5 * width, center_freq + 5 * width, num_points
        )
        profile = (
            amplitude
            * (width / 2) ** 2
            / ((freq_range - center_freq) ** 2 + (width / 2) ** 2)
        )
        return freq_range, profile


class SpectroscopyWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.analyzer = SpectroscopyAnalyzer()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Create figure for spectral plot
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)

        # Create group box
        group_box = QGroupBox("Spectral Analysis")
        group_layout = QVBoxLayout()
        group_layout.addWidget(self.canvas)
        group_box.setLayout(group_layout)

        layout.addWidget(group_box)
        self.setLayout(layout)

    def update_plot(self, frequencies, amplitudes):
        """Update spectral plot with new data"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        spectral_data = self.analyzer.analyze_spectrum(frequencies, amplitudes)

        # Plot individual spectral lines
        for data in spectral_data:
            freq_range, profile = data["line_profile"]
            ax.plot(freq_range, profile, alpha=0.7)

            # Annotate atomic transitions
            for transition in data["transitions"][:2]:  # Show top 2 matches
                ax.annotate(
                    f"{transition['element']} {transition['transition']}",
                    xy=(data["frequency"], data["amplitude"]),
                    xytext=(10, 10),
                    textcoords="offset points",
                )

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Intensity")
        ax.set_title("Spectral Analysis")
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()
