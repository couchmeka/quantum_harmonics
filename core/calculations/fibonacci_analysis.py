import numpy as np


class FibonacciAnalyzer:
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    def generate_fibonacci_sequence(self, n_terms=30):
        """Generate Fibonacci sequence up to n terms"""
        sequence = [0, 1]
        for i in range(2, n_terms):
            sequence.append(sequence[i - 1] + sequence[i - 2])
        return np.array(sequence)

    def scale_with_phi(self, sequence):
        """Scale Fibonacci numbers by Phi"""
        return sequence * self.phi

    def find_musical_matches(
        self, fibonacci_frequencies, musical_frequencies, tolerance=0.05
    ):
        """Find matches between Fibonacci-derived frequencies and musical frequencies"""
        matches = []
        for fib_freq in fibonacci_frequencies:
            for musical_freq in musical_frequencies:
                deviation = abs(fib_freq - musical_freq) / musical_freq
                if deviation < tolerance:
                    matches.append(
                        {
                            "fibonacci_freq": fib_freq,
                            "musical_freq": musical_freq,
                            "deviation": deviation,
                        }
                    )
        return matches

    def analyze_quantum_correlation(self, matches, statevector=None):
        """Analyze quantum correlation between Fibonacci and musical frequencies"""
        if statevector is None:
            # Generate default statevector if none provided
            freqs = np.array([match["musical_freq"] for match in matches])
            if len(freqs) > 0:
                statevector = freqs / np.sqrt(np.sum(freqs**2))
            else:
                return None

        correlations = []
        for match in matches:
            fib_freq = match["fibonacci_freq"]
            musical_freq = match["musical_freq"]
            # Calculate quantum correlation using statevector
            correlation = np.abs(
                np.vdot(statevector, np.exp(1j * 2 * np.pi * fib_freq / musical_freq))
            )
            correlations.append(
                {
                    "fibonacci_freq": fib_freq,
                    "musical_freq": musical_freq,
                    "quantum_correlation": correlation,
                }
            )

        return correlations


class QuantumStateManager:
    @staticmethod
    def validate_statevector(frequencies, amplitudes=None):
        """Validate and generate statevector from frequencies"""
        if not frequencies:
            return None

        if amplitudes is None:
            amplitudes = np.ones(len(frequencies))

        # Normalize to create valid quantum state
        norm = np.sqrt(np.sum(np.abs(amplitudes) ** 2))
        if norm == 0:
            return None

        return amplitudes / norm

    @staticmethod
    def calculate_quantum_metrics(statevector):
        """Calculate quantum metrics for the statevector"""
        if statevector is None:
            return {"purity": 0.0, "fidelity": 0.0, "phases": []}

        # Calculate density matrix
        density_matrix = np.outer(statevector, np.conjugate(statevector))

        # Calculate purity
        purity = np.abs(np.trace(np.matmul(density_matrix, density_matrix)))

        # Calculate fidelity with respect to pure state
        fidelity = np.abs(np.vdot(statevector, statevector))

        # Extract phases
        phases = np.angle(statevector)

        return {"purity": purity, "fidelity": fidelity, "phases": phases.tolist()}


def update_quantum_data_manager(data_manager, fibonacci_results, quantum_results):
    """Update data manager with Fibonacci and quantum analysis results"""
    combined_results = {
        "fibonacci_analysis": fibonacci_results,
        "quantum_frequencies": quantum_results.get("frequencies", []),
        "statevector": quantum_results.get("statevector", []),
        "quantum_metrics": quantum_results.get("metrics", {}),
        "correlations": fibonacci_results.get("correlations", []),
    }

    # Update both melody and quantum results
    data_manager.update_melody_results(combined_results)
    data_manager.update_quantum_results(combined_results)

    return combined_results
