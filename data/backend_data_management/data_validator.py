# data_validator.py
import traceback
from datetime import datetime

import numpy as np
from typing import Dict, Optional, List


class DataValidator:
    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio constant

    @staticmethod
    def validate_frequencies(
        data: Dict, key_name: str = "quantum_frequencies"
    ) -> Optional[np.ndarray]:
        """Validate and standardize frequency data"""
        if not isinstance(data, dict):
            return None

        frequencies = data.get(key_name, [])
        if isinstance(frequencies, list):
            frequencies = np.array(frequencies)
        elif not isinstance(frequencies, np.ndarray):
            return None

        if len(frequencies) == 0:
            return None

        return frequencies

    @staticmethod
    def validate_statevector(data: Dict) -> Optional[np.ndarray]:
        """Validate and normalize statevector data"""
        if not isinstance(data, dict):
            return None

        # Direct access first
        if "statevector" in data:
            try:
                statevector = np.array(data["statevector"])
                if len(statevector) > 0:
                    # Normalize if needed
                    norm = np.linalg.norm(statevector)
                    if norm > 0:
                        return statevector / norm
            except Exception as e:
                print(f"Error processing direct statevector: {str(e)}")

        # Try generating from frequencies and amplitudes
        try:
            frequencies = data.get("quantum_frequencies", [])
            amplitudes = data.get("amplitudes", [])

            if len(frequencies) > 0:
                if len(amplitudes) == 0:
                    amplitudes = frequencies

                # Match lengths
                min_len = min(len(frequencies), len(amplitudes))
                frequencies = frequencies[:min_len]
                amplitudes = amplitudes[:min_len]

                # Create statevector
                statevector = np.array(amplitudes) * np.exp(1j * np.angle(frequencies))
                norm = np.linalg.norm(statevector)
                if norm > 0:
                    return statevector / norm
        except Exception as e:
            print(f"Error generating statevector: {str(e)}")

        return None

    @staticmethod
    def generate_fibonacci_frequencies(n_terms: int = 30) -> np.ndarray:
        """Generate Fibonacci sequence scaled by golden ratio"""
        fib = [0, 1]
        for i in range(2, n_terms):
            fib.append(fib[i - 1] + fib[i - 2])
        return np.array(fib) * DataValidator.PHI

    @staticmethod
    def find_musical_matches(
        fibonacci_frequencies: np.ndarray,
        musical_frequencies: np.ndarray,
        tolerance: float = 0.05,
    ) -> List[Dict]:
        """Find matches between Fibonacci-derived and musical frequencies"""
        matches = []
        for fib_freq in fibonacci_frequencies:
            for musical_freq in musical_frequencies:
                # Skip zero frequency to avoid division by zero
                if musical_freq == 0:
                    continue

                try:
                    deviation = abs(fib_freq - musical_freq) / musical_freq
                    if deviation < tolerance:
                        matches.append(
                            {
                                "fibonacci_freq": float(fib_freq),
                                "musical_freq": float(musical_freq),
                                "deviation": float(deviation),
                            }
                        )
                except ZeroDivisionError:
                    continue
                except Exception as e:
                    print(f"Error calculating deviation: {str(e)}")
                    continue

        return matches

    @staticmethod
    def validate_quantum_data(
        data: Dict, required_keys: List[str] = None
    ) -> Optional[Dict]:
        """
        Validate quantum analysis data with improved error checking and logging.
        """
        if not data:
            print("Validation failed: No data provided")
            return None

        validated = {}

        try:
            # Validate specific keys if provided
            if required_keys:
                for key in required_keys:
                    if key not in data:
                        print(f"Validation failed: Missing key '{key}' in data")
                        continue

                    value = data[key]
                    print(f"Validating {key}: {type(value)}")

                    # Handle NumPy arrays safely
                    if isinstance(value, np.ndarray):
                        value = value.tolist()

                    # Validate based on presence and non-emptiness
                    if value is None or (isinstance(value, list) and not value):
                        print(f"Validation failed: Key '{key}' is empty")
                    else:
                        validated[key] = value

            # Validate the entire dataset if no specific keys are provided
            else:
                for key, value in data.items():
                    print(f"Validating {key}: {type(value)}")

                    # Handle NumPy arrays safely
                    if isinstance(value, np.ndarray):
                        value = value.tolist()

                    # Validate based on presence and non-emptiness
                    if value is None or (isinstance(value, list) and not value):
                        print(f"Warning: Key '{key}' is missing or empty")
                    else:
                        validated[key] = value

            if validated:
                return validated

            print("Validation failed: No valid quantum data found")
            return None

        except Exception as e:
            print(f"Error during validation: {str(e)}")
            traceback.print_exc()
            return None

    @staticmethod
    def validate_qec_data(data):
        if not isinstance(data, dict):
            return None

        # Extract QEC-specific data with material details
        qec_data = {
            "metrics": data.get("metrics", {}),
            "qec_type": data.get("qec_type", ""),
            "material": data.get("material", {}),
            "initial_fidelity": data.get("initial_fidelity", 0.0),
            "final_fidelity": data.get("final_fidelity", 0.0),
            "improvement": data.get("improvement", 0.0),
        }

        # Preserve material name and properties if present
        material_data = data.get("material", {})
        if isinstance(material_data, dict):
            qec_data["material"] = {
                "name": material_data.get("name", "Unknown"),
                "temperature": material_data.get("temperature", 0),
                "properties": material_data.get("properties", {}),
            }

        # Log validation for debugging
        print(f"QEC Validation - Material data: {qec_data['material']}")

        # Validate at least some data exists
        if not any(qec_data.values()) or not qec_data.get("material", {}).get("name"):
            print("Warning: Missing QEC or material data")
            return None

        return qec_data

    @staticmethod
    def validate_particle_data(data: Dict) -> Optional[Dict]:
        """
        Validate particle simulation data
        """
        validated_quantum = DataValidator.validate_quantum_data(data)
        if validated_quantum is None:
            return None

        return {
            **validated_quantum,
            "positions": data.get("positions", []),
            "velocities": data.get("velocities", []),
            "mode": data.get("mode", ""),
            "analysis_type": "particle",
        }

    @staticmethod
    def validate_fluid_data(data: Dict) -> Optional[Dict]:
        """
        Validate fluid dynamics data to ensure required keys and data structure.

        Args:
            data (Dict): Input data to validate.

        Returns:
            Optional[Dict]: Validated fluid data or None if validation fails.
        """
        if not isinstance(data, dict):
            print("Validation failed: Fluid data is not a dictionary.")
            return None

        required_keys = [
            "original_frequencies",
            "fibonacci_frequencies",
            "solution",
            "t",
        ]
        validated_data = {}

        for key in required_keys:
            if key not in data:
                print(f"Validation failed: Missing required key '{key}'.")
                continue

            value = data[key]
            if isinstance(value, list):
                value = np.array(value)  # Convert lists to NumPy arrays
            if not isinstance(value, (np.ndarray, list)) or len(value) == 0:
                print(f"Validation failed: Key '{key}' is empty or invalid.")
                continue

            validated_data[key] = value

        if all(key in validated_data for key in required_keys):
            print("Fluid data validation succeeded.")
            return validated_data

        print("Validation failed: Incomplete or invalid fluid data.")
        return None


def validate_and_update(manager, data: Dict, analysis_type: str) -> bool:
    """
    Validate and update data manager with the correct data type

    Args:
        manager: QuantumDataManager instance
        data: Data to validate and store
        analysis_type: Type of analysis ('quantum', 'qec', or 'particle')

    Returns:
        bool: True if update successful, False otherwise
    """
    if analysis_type == "quantum":
        validated = DataValidator.validate_quantum_data(data)
        if validated:
            manager.update_quantum_results(validated)
            return True
    elif analysis_type == "qec":
        validated = DataValidator.validate_qec_data(data)
        if validated:
            manager.update_qec_results(validated)
            return True
    elif analysis_type == "particle":
        validated = DataValidator.validate_particle_data(data)
        if validated:
            manager.update_particle_results(validated)
            return True
    return False
