# data_handler.py
from datetime import datetime


class StandardizedDataHandler:
    @staticmethod
    def format_quantum_results(results):
        """Standardize quantum analysis results"""
        if not results:
            return None

        return {
            "quantum_metrics": {
                "purity": results.get("purity", 0.0),
                "fidelity": results.get("fidelity", 0.0),
            },
            "frequencies": results.get("frequencies", []),
            "amplitudes": results.get("amplitudes", []),
            "statevector": results.get("statevector", []),
            "phases": results.get("phases", []),
            "analysis_type": "quantum",
        }

    @staticmethod
    def format_melody_results(results):
        """Standardize melody analysis results"""
        if not results:
            return None

        return {
            "notes": results.get("notes", []),
            "frequencies": results.get("frequencies", []),
            "musical_systems": results.get("musical_systems", {}),
            "analysis_type": "melody",
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
        """Standardize QEC analysis results"""
        if not results:
            return None

        metrics = results.get("metrics", {})
        return {
            "fidelity": {
                "initial": metrics.get("initial_fidelity", 0.0),
                "final": metrics.get("final_fidelity", 0.0),
            },
            "improvement": metrics.get("improvement", 0.0),
            "material_impact": metrics.get("material_impact", 0.0),
            "qec_type": results.get("params", {}).get("qec_type", ""),
            "analysis_type": "qec",
        }

    @staticmethod
    def combine_results(quantum=None, melody=None, fluid=None, qec=None):
        """Combine all results into a standardized format"""
        combined = {"timestamp": datetime.now().isoformat(), "analysis_results": {}}

        if quantum:
            combined["analysis_results"]["quantum"] = (
                StandardizedDataHandler.format_quantum_results(quantum)
            )
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

        return combined
