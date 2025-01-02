# quantum_results_interpreter.py,
# contains the QuantumResultsInterpreter class, which is designed to interpret
# and analyze quantum computing results. The class integrates a data manager to
# handle data operations and provides methods to format, interpret, and summarize quantum metrics and results.
# It aims to present complex quantum data in a human-readable format, making it
# accessible for analysis and insights. The class handles various types of quantum data,
# including quantum states, melody harmonics, and quantum error correction results,
# providing a comprehensive tool for quantum data analysis.


import numpy as np
from storage.data_manager import QuantumDataManager


class QuantumResultsInterpreter:
    def __init__(self):
        self.data_manager = QuantumDataManager()

    def format_metric(self, value, format_str=".3f"):
        """Safely format a metric value"""
        try:
            if isinstance(value, (float, int)):
                return f"{float(value):{format_str}}"
            if value == "N/A" or value is None:
                return "N/A"
            # Try to convert to float if it's a string representing a number
            try:
                return f"{float(value):{format_str}}"
            except (ValueError, TypeError):
                return str(value)
        except Exception as e:
            print(f"Error formatting metric: {str(e)}")
            return "N/A"

    def interpret_results(self, results):
        """Interpret quantum results in a human-readable format"""
        if not results:
            return "No quantum data available for interpretation."

        try:
            interpretations = []

            # Quantum Results Interpretation
            if "quantum_results" in results:
                quantum_results = results["quantum_results"]
                interpretations.append("Quantum Analysis Insights:")
                interpretations.append(
                    f"• Frequencies Analyzed: {len(quantum_results.get('frequencies', []))}"
                )

                purity = quantum_results.get("purity", "N/A")
                fidelity = quantum_results.get("fidelity", "N/A")

                interpretations.append(f"• State Purity: {self.format_metric(purity)}")
                interpretations.append(
                    f"• Quantum Fidelity: {self.format_metric(fidelity)}"
                )

            # Melody Results Interpretation
            if "melody_results" in results:
                melody_results = results["melody_results"]
                interpretations.append("\nMelody Quantum Harmonics:")
                interpretations.append(
                    f"• Notes Analyzed: {', '.join(str(n) for n in melody_results.get('notes', []))}"
                )

                frequencies = []
                for f in melody_results.get("frequencies", []):
                    try:
                        frequencies.append(f"{float(f):.2f} Hz")
                    except (ValueError, TypeError):
                        frequencies.append(str(f))
                interpretations.append(f"• Frequencies: {frequencies}")

            # QEC Results Interpretation
            if "qec_results" in results:
                qec_results = results["qec_results"]
                interpretations.append("\nQuantum Error Correction Analysis:")
                metrics = qec_results.get("metrics", {})

                initial_fid = metrics.get("initial_fidelity", "N/A")
                final_fid = metrics.get("final_fidelity", "N/A")
                improvement = metrics.get("improvement", "N/A")

                interpretations.append(
                    f"• Initial Fidelity: {self.format_metric(initial_fid)}"
                )
                interpretations.append(
                    f"• Final Fidelity: {self.format_metric(final_fid)}"
                )

                # Special handling for improvement percentage
                if isinstance(improvement, (float, int)):
                    interpretations.append(
                        f"• Performance Improvement: {improvement:+.1f}%"
                    )
                else:
                    interpretations.append(f"• Performance Improvement: N/A")

            return "\n".join(interpretations)

        except Exception as e:
            print(f"Error interpreting results: {str(e)}")
            return "Error occurred while interpreting quantum results."

    def generate_summary(self, results):
        """Generate a concise summary of quantum results"""
        try:
            summary = []

            if results:
                summary.append("Comprehensive Quantum Systems Analysis:")

                # Check data sources and number of analyses
                data_sources = [
                    ("Quantum", results.get("quantum_results")),
                    ("Melody", results.get("melody_results")),
                    ("QEC", results.get("qec_results")),
                ]

                active_sources = [name for name, data in data_sources if data]
                if active_sources:
                    summary.append(
                        f"• Active Data Sources: {', '.join(active_sources)}"
                    )
                else:
                    summary.append("• No active data sources found")

                # Safely calculate average purity
                try:
                    purities = []
                    quantum_purity = results.get("quantum_results", {}).get("purity")
                    if isinstance(quantum_purity, (float, int)):
                        purities.append(quantum_purity)

                    qec_fidelity = (
                        results.get("qec_results", {})
                        .get("metrics", {})
                        .get("initial_fidelity")
                    )
                    if isinstance(qec_fidelity, (float, int)):
                        purities.append(qec_fidelity)

                    if purities:
                        avg_purity = np.mean(purities)
                        summary.append(f"• Overall System Coherence: {avg_purity:.3f}")
                    else:
                        summary.append("• Overall System Coherence: Not available")
                except Exception as e:
                    print(f"Error calculating coherence: {str(e)}")
                    summary.append("• Overall System Coherence: Not available")
            else:
                summary.append("No quantum data available for analysis.")

            return "\n".join(summary)

        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return "Error occurred while generating quantum summary."
