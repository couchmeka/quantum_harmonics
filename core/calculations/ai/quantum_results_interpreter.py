# quantum_results_interpreter.py,
# contains the QuantumResultsInterpreter class, which is designed to interpret
# and analyze quantum computing results. The class integrates a data manager to
# handle data operations and provides methods to format, interpret, and summarize quantum metrics and results.
# It aims to present complex quantum data in a human-readable format, making it
# accessible for analysis and insights. The class handles various types of quantum data,
# including quantum states, melody harmonics, and quantum error correction results,
# providing a comprehensive tool for quantum data analysis.


import numpy as np
from data.backend_data_management.data_manager import QuantumDataManager


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

            for system_name, system_data in results.items():
                if not system_data:
                    continue

                interpretations.append(f"\n{system_name.upper()} Analysis:")

                # Quantum System
                if system_name == "quantum":
                    frequencies = system_data.get("quantum_frequencies", [])
                    interpretations.append(
                        f"• Frequencies Analyzed: {len(frequencies)}"
                    )
                    purity = system_data.get("purity", "N/A")
                    fidelity = system_data.get("fidelity", "N/A")
                    interpretations.append(
                        f"• State Purity: {self.format_metric(purity)}"
                    )
                    interpretations.append(
                        f"• Quantum Fidelity: {self.format_metric(fidelity)}"
                    )

                # Melody System
                elif system_name == "melody":
                    notes = system_data.get("notes", [])
                    frequencies = system_data.get("frequencies", [])
                    interpretations.append(f"• Notes Analyzed: {len(notes)}")
                    interpretations.append(f"• Frequencies: {len(frequencies)}")

                # Fluid System
                elif system_name == "fluid":
                    orig_freq = system_data.get("original_frequencies", [])
                    fib_freq = system_data.get("fibonacci_frequencies", [])
                    interpretations.append(f"• Original Frequencies: {len(orig_freq)}")
                    interpretations.append(f"• Fibonacci Frequencies: {len(fib_freq)}")
                    if orig_freq:
                        interpretations.append(
                            f"• Frequency Range: {min(orig_freq):.1f} - {max(orig_freq):.1f} Hz"
                        )

                # QEC System
                elif system_name == "qec":
                    metrics = system_data.get("metrics", {})
                    initial_fid = metrics.get("initial_fidelity", "N/A")
                    final_fid = metrics.get("final_fidelity", "N/A")
                    improvement = metrics.get("improvement", "N/A")
                    interpretations.append(
                        f"• Initial Fidelity: {self.format_metric(initial_fid)}"
                    )
                    interpretations.append(
                        f"• Final Fidelity: {self.format_metric(final_fid)}"
                    )
                    if isinstance(improvement, (float, int)):
                        interpretations.append(
                            f"• Performance Improvement: {improvement:+.1f}%"
                        )

                # Particle System
                elif system_name == "particle":
                    positions = system_data.get("positions", [])
                    velocities = system_data.get("velocities", [])
                    interpretations.append(f"• Number of Particles: {len(positions)}")
                    interpretations.append(f"• Velocity Data Points: {len(velocities)}")

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

                # Check active systems dynamically
                active_sources = [name for name, data in results.items() if data]
                if active_sources:
                    summary.append(
                        f"• Active Data Sources: {', '.join(active_sources)}"
                    )
                else:
                    summary.append("• No active data sources found")

                # Calculate overall coherence
                try:
                    purities = [
                        data.get("purity")
                        for system_name, data in results.items()
                        if data and "purity" in data
                    ]
                    purities = [p for p in purities if isinstance(p, (float, int))]
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
