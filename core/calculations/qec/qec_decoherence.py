import numpy as np
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit import transpile
from qiskit_aer import AerSimulator


class QECDecoherenceHandler:
    def __init__(self):
        self.noise_model = None
        self.simulator = AerSimulator()

    def create_decoherence_model(self, t1, t2):
        """Create a noise model with T1/T2 relaxation"""
        self.noise_model = NoiseModel()
        error = thermal_relaxation_error(t1, t2, 0)
        self.noise_model.add_all_qubit_quantum_error(error, ["u1", "u2", "u3"])
        return self.noise_model

    @staticmethod
    def calculate_improvement(initial_fidelity, final_fidelity):
        """Calculate improvement percentage in fidelity"""
        return (final_fidelity / initial_fidelity - 1) * 100

    @staticmethod
    def theoretical_decay(time_points, t2):
        """Calculate theoretical T2 decay"""
        return np.exp(-time_points / t2)

    def simulate_decoherence(self, circuit, t1, t2):
        """Run decoherence simulation"""
        self.create_decoherence_model(t1, t2)

        # Transpile circuit for simulator
        transpiled = transpile(circuit, self.simulator)

        # Run simulation
        time_points = np.linspace(0, max(t1, t2), 20)
        results = []

        for t in time_points:
            # Run circuit with current noise model
            job = self.simulator.run(
                transpiled, noise_model=self.noise_model, shots=1000
            )
            counts = job.result().get_counts()

            # Calculate fidelity
            total = sum(counts.values())
            correct_count = counts.get("0" * circuit.num_qubits, 0)
            fidelity = correct_count / total

            results.append(
                {
                    "time": t,
                    "fidelity": fidelity,
                    "counts": counts,
                    "theoretical_decay": self.theoretical_decay(t, t2),
                }
            )

        return {
            "time_points": time_points,
            "results": results,
            "params": {"t1": t1, "t2": t2},
        }
