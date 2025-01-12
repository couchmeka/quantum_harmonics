import numpy as np

from .qec_circuit_builder import QECCircuitBuilder
from .qec_decoherence import QECDecoherenceHandler


class QECAnalyzer:
    def __init__(self):
        self.circuit_builder = QECCircuitBuilder()
        self.decoherence = QECDecoherenceHandler()

    def analyze_circuit(self, qec_type, t1, t2, material_properties=None):
        try:
            # Base error rates
            error_rate = 1 / (t1 * 0.001)  # Convert ms to s
            dephasing_rate = 1 / (t2 * 0.001)  # Convert ms to s

            # Much stronger temperature effects
            material_factor = 1.0
            if material_properties:
                # Temperature now has a much stronger effect
                temp = material_properties.get("temperature", 300)
                # Exponential temperature dependence instead of quadratic
                thermal_factor = np.exp(
                    (temp - 300) / 100
                )  # Much stronger temperature scaling

                # Material-specific effects
                phonon_coupling = material_properties.get("debye_freq", 0) * 1e-11
                density_factor = material_properties.get("density", 1000) / 1000

                # Combined material effects with emphasis on temperature
                material_factor = (
                    thermal_factor * (1 + phonon_coupling) * density_factor
                )

                # Scale error rates more strongly with temperature
                error_rate *= material_factor * (temp / 300)
                dephasing_rate *= (
                    material_factor * (temp / 300) ** 2
                )  # Dephasing more sensitive to temperature

            # Time points in milliseconds
            time_points = np.linspace(0, t1, 100)
            results = []

            for t in time_points:
                t_seconds = t * 0.001

                # Environmental effects now temperature dependent
                env_factor = 1 + (t_seconds / 5)
                if material_properties:
                    env_factor *= (
                        temp / 300
                    )  # Environment more active at higher temperatures

                # Calculate error probabilities
                bit_flip_prob = 1 - np.exp(-error_rate * t_seconds * env_factor)
                phase_flip_prob = 1 - np.exp(-dephasing_rate * t_seconds * env_factor)

                # QEC behavior now temperature dependent
                if qec_type == "3-Qubit Bit Flip":
                    base_efficiency = 0.95 / (
                        1 + (temp - 300) / 500
                    )  # Efficiency decreases with temperature
                    qec_fidelity = 1 - (bit_flip_prob**2 * 0.05 + phase_flip_prob * 0.5)

                elif qec_type == "3-Qubit Phase Flip":
                    base_efficiency = 0.93 / (1 + (temp - 300) / 500)
                    qec_fidelity = 1 - (phase_flip_prob**2 * 0.05 + bit_flip_prob * 0.5)

                elif qec_type == "5-Qubit Code":
                    base_efficiency = 0.90 / (1 + (temp - 300) / 600)
                    qec_fidelity = 1 - (bit_flip_prob**2 + phase_flip_prob**2) * 0.1

                elif qec_type == "7-Qubit Steane":
                    base_efficiency = 0.92 / (1 + (temp - 300) / 700)
                    qec_fidelity = 1 - (bit_flip_prob + phase_flip_prob) * 0.08

                else:  # 9-Qubit Shor
                    base_efficiency = 0.88 / (1 + (temp - 300) / 800)
                    qec_fidelity = 1 - (bit_flip_prob + phase_flip_prob) * 0.12

                # Time and temperature dependent efficiency decay
                if material_properties:
                    efficiency = base_efficiency * (1 - t_seconds * (temp / 300) / 10)
                else:
                    efficiency = base_efficiency * (1 - t_seconds / 10)
                qec_fidelity *= efficiency

                # More rapid decay without QEC, temperature dependent
                no_qec_factor = 2
                if material_properties:
                    no_qec_factor *= temp / 300
                no_qec_fidelity = np.exp(
                    -(error_rate + dephasing_rate)
                    * t_seconds
                    * env_factor
                    * no_qec_factor
                )

                # Ensure fidelities stay in valid range
                qec_fidelity = max(0, min(1, qec_fidelity))
                no_qec_fidelity = max(0, min(1, no_qec_fidelity))

                # State distribution
                total_counts = 1000
                error_counts = int((1 - qec_fidelity) * total_counts)
                counts = {"000": total_counts - error_counts, "111": error_counts}

                results.append(
                    {
                        "time": t,
                        "fidelity": qec_fidelity,
                        "theoretical_decay": no_qec_fidelity,
                        "counts": counts,
                        "error_components": {
                            "bit_flip": bit_flip_prob,
                            "phase_flip": phase_flip_prob,
                            "efficiency": efficiency,
                            "material_factor": material_factor,
                            "temperature": temp if material_properties else 300,
                        },
                    }
                )

            initial_fidelity = results[0]["fidelity"]
            final_fidelity = results[-1]["fidelity"]
            theoretical_final = results[-1]["theoretical_decay"]
            improvement = 0.0
            if theoretical_final > 0:
                improvement = (
                    (final_fidelity - theoretical_final) / theoretical_final
                ) * 100

            return {
                "simulation_results": {"time_points": time_points, "results": results},
                "metrics": {
                    "initial_fidelity": initial_fidelity,
                    "final_fidelity": final_fidelity,
                    "improvement": improvement,
                    "material_impact": material_factor,
                    "temperature_impact": (
                        thermal_factor if material_properties else 1.0
                    ),
                },
                "params": {
                    "t1": t1,
                    "t2": t2,
                    "qec_type": qec_type,
                    "temperature": temp if material_properties else 300,
                },
            }

        except Exception as e:
            print(f"Error in QEC analysis: {str(e)}")
            raise

    def calculate_improvement(self, final_fidelity, theoretical_final):
        """Calculate improvement percentage safely"""
        if theoretical_final == 0 or theoretical_final is None:
            return 0.0
        try:
            improvement = (
                (final_fidelity - theoretical_final) / theoretical_final
            ) * 100
            return improvement
        except (ZeroDivisionError, TypeError):
            return 0.0

    def _get_circuit(self, qec_type):
        """Get the appropriate QEC circuit based on type"""
        if qec_type == "3-Qubit Bit Flip":
            return self.circuit_builder.create_3qubit_bit_flip()
        elif qec_type == "3-Qubit Phase Flip":
            return self.circuit_builder.create_3qubit_phase_flip()
        elif qec_type == "5-Qubit Code":
            return self.circuit_builder.create_5qubit_code()
        elif qec_type == "7-Qubit Steane":
            return self.circuit_builder.create_steane_code()
        elif qec_type == "9-Qubit Shor":
            return self.circuit_builder.create_shor_code()
        else:
            raise ValueError(f"Unknown QEC type: {qec_type}")
