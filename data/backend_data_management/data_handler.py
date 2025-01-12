# data_handler.py
from datetime import datetime

from data.backend_data_management.data_validator import DataValidator


class StandardizedDataHandler:
    @staticmethod
    def format_quantum_results(results):
        return DataValidator.validate_quantum_data(results)

    @staticmethod
    def format_melody_results(results):
        """Standardize melody analysis results"""
        if not results:
            return None

        return {
            "notes": results.get("notes", []),
            "frequencies": results.get("frequencies", []),
            "musical_systems": results.get("musical_systems", {}),
            "eigenvalues": results.get("eigenvalues", []),
            "wave_solution": results.get("wave_solution", []),
            "t": results.get("t", []),
            "analysis_type": "melody",
            "note_combinations": results.get("note_combinations", {}),
            "harmonic_ratios": results.get("harmonic_ratios", {}),
        }

    @staticmethod
    def format_fluid_results(results):
        """Standardize fluid dynamics results"""
        if not results:
            return None

        return {
            "original_frequencies": results.get("original_frequencies", []),
            "fibonacci_frequencies": results.get("fibonacci_frequencies", []),
            "solution": results.get("solution", []),
            "t": results.get("t", []),
            "analysis_type": "fluid",
        }

    @staticmethod
    def format_qec_results(results):
        return DataValidator.validate_qec_data(results)

    @staticmethod
    def format_particle_results(results):
        return DataValidator.validate_particle_data(results)

    @staticmethod
    def combine_results(quantum=None, melody=None, fluid=None, qec=None, particle=None):
        """Combine all results into a standardized format"""
        combined = {"timestamp": datetime.now().isoformat(), "analysis_results": {}}

        if quantum:
            quantum_data = StandardizedDataHandler.format_quantum_results(quantum)
            if quantum_data:
                quantum_data["quantum_frequencies"] = quantum_data.pop(
                    "frequencies", []
                )
                combined["analysis_results"]["quantum"] = quantum_data

        if melody:
            combined["analysis_results"]["melody"] = (
                StandardizedDataHandler.format_melody_results(melody)
            )
        if fluid:
            combined["analysis_results"]["fluid"] = (
                StandardizedDataHandler.format_fluid_results(fluid)
            )
        if qec:
            combined["analysis_results"]["qec"] = (
                StandardizedDataHandler.format_qec_results(qec)
            )
        if particle:
            combined["analysis_results"]["particle"] = (
                StandardizedDataHandler.format_particle_results(particle)
            )
            print(f"Combined Results: {combined}")

        return combined
