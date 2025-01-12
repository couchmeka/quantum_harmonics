# core/calculations/quantum/pythagorean_analysis.py
# This module provides analysis of quantum harmonics using Pythagorean ratios.
# It calculates relationships between frequencies based on traditional
# Pythagorean musical intervals and their quantum mechanical influences.
#
# Key Components:
# - Pythagorean ratio definitions and calculations
# - Harmonic frequency analysis
# - Quantum influence calculations for frequency ratios
# - Phase alignment and coherence factor analysis

import numpy as np


class PythagoreanAnalyzer:
    def __init__(self):
        # Define standard Pythagorean ratios for musical intervals
        self.pythagorean_ratios = {
            "unison": 1 / 1,
            "octave": 2 / 1,
            "perfect_fifth": 3 / 2,
            "perfect_fourth": 4 / 3,
            "major_third": 81 / 64,
            "minor_third": 32 / 27,
            "major_second": 9 / 8,
            "minor_second": 256 / 243,
        }

    def analyze_harmonics(self, frequencies):
        """
        Analyze frequencies using Pythagorean ratios.

        Args:
            frequencies (list): List of frequencies to analyze

        Returns:
            list: Analysis results including ratios and deviations
        """
        base_freq = frequencies[0]
        analysis = []

        for freq in frequencies:
            if freq == 0:
                print(f"Skipping frequency {freq} because it is zero.")
                continue

        for freq in frequencies:
            ratio = freq / base_freq
            closest_pythagorean = self._find_closest_ratio(ratio)
            deviation = abs(ratio - closest_pythagorean["ratio"])

            # Calculate harmonic series influence
            harmonic_influence = sum(
                1 / n for n in range(1, 9) if abs(freq / (base_freq * n) - 1) < 0.1
            )

            analysis.append(
                {
                    "frequency": freq,
                    "ratio": ratio,
                    "closest_interval": closest_pythagorean["name"],
                    "pythagorean_ratio": closest_pythagorean["ratio"],
                    "deviation": deviation,
                    "harmonic_influence": harmonic_influence,
                }
            )

        return analysis

    def _find_closest_ratio(self, target_ratio):
        """
        Find the closest Pythagorean ratio to the target.

        Args:
            target_ratio (float): Ratio to match

        Returns:
            dict: Name and value of closest Pythagorean ratio
        """
        closest = min(
            self.pythagorean_ratios.items(),
            key=lambda x: abs(float(x[1]) - target_ratio),
        )
        return {"name": closest[0], "ratio": closest[1]}

    def calculate_quantum_influence(self, analysis_results):
        """
        Calculate quantum mechanical influence of Pythagorean ratios.

        Args:
            analysis_results (list): Results from analyze_harmonics

        Returns:
            list: Quantum influence calculations for each frequency
        """
        quantum_influences = []

        for result in analysis_results:
            # Calculate phase alignment based on ratio deviation
            phase_alignment = np.exp(-result["deviation"] * np.pi)

            # Calculate coherence influence based on harmonic content
            coherence_factor = (
                result["harmonic_influence"] / 8
            )  # Normalize by max harmonics

            # Calculate overall quantum influence
            quantum_influence = {
                "frequency": result["frequency"],
                "phase_alignment": phase_alignment,
                "coherence_factor": coherence_factor,
                "total_influence": (phase_alignment + coherence_factor) / 2,
            }

            quantum_influences.append(quantum_influence)

        return quantum_influences
