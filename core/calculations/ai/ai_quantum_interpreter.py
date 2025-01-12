# ai_quantum_interpreter.py,
# developed to use advanced AI models for
# interpreting and analyzing quantum computing data.
# It integrates various functionalities for managing quantum data,
# analyzing quantum states, and providing insights into quantum measurements and operations.
#
# import os
# import numpy as np
# from anthropic import Anthropic
# from data.data_manager import QuantumDataManager
# from core.calculations.quantum.quantum_harmonic_analysis import QuantumHarmonicsAnalyzer
#
#
# class QuantumInterpreter:
#     def __init__(self, model_name="claude-3-opus-20240229"):
#         self.model_name = model_name
#         self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
#         if not os.getenv("ANTHROPIC_API_KEY"):
#             raise ValueError("Anthropic API key not found in environment variables")
#         self.data_manager = QuantumDataManager()
#         self.analyzer = QuantumHarmonicsAnalyzer()
#
#     def _format_metric(self, value, default="N/A"):
#         """Safely format a metric value"""
#         if isinstance(value, (float, int)):
#             return f"{value:.4f}"
#         return str(default)
#
#     def generate_summary(self, results):
#         """Generate a concise summary of quantum results"""
#         try:
#             summary = []
#
#             if results:
#                 summary.append("Comprehensive Quantum Systems Analysis:")
#
#                 # Check data sources and number of analyses
#                 data_sources = [
#                     ("Quantum", results.get("quantum_results")),
#                     ("Melody", results.get("melody_results")),
#                     ("QEC", results.get("qec_results")),
#                     ("Particle", results.get("particle_results")),
#                     ("Fluid", results.get("fluid_results")),
#                 ]
#
#                 active_sources = [name for name, data in data_sources if data]
#                 if active_sources:
#                     summary.append(
#                         f"• Active Data Sources: {', '.join(active_sources)}"
#                     )
#                 else:
#                     summary.append("• No active data sources found")
#
#                 # Safely calculate average purity
#                 try:
#                     purities = []
#                     quantum_purity = results.get("quantum_results", {}).get("purity")
#                     if isinstance(quantum_purity, (float, int)):
#                         purities.append(quantum_purity)
#
#                     qec_fidelity = (
#                         results.get("qec_results", {})
#                         .get("metrics", {})
#                         .get("initial_fidelity")
#                     )
#                     if isinstance(qec_fidelity, (float, int)):
#                         purities.append(qec_fidelity)
#
#                     if purities:
#                         avg_purity = np.mean(purities)
#                         summary.append(f"• Overall System Coherence: {avg_purity:.3f}")
#                     else:
#                         summary.append("• Overall System Coherence: Not available")
#                 except Exception as e:
#                     print(f"Error calculating coherence: {str(e)}")
#                     summary.append("• Overall System Coherence: Not available")
#             else:
#                 summary.append("No quantum data available for analysis.")
#
#             return "\n".join(summary)
#
#         except Exception as e:
#             print(f"Error generating summary: {str(e)}")
#             return "Error occurred while generating quantum summary."
#
#     def interpret_quantum_data(self, query=None):
#         try:
#             # Fetch the latest results from the data manager
#             all_results = self.data_manager.get_latest_results()
#
#             # Combine all results into a single dictionary for analysis
#             combined_results = {}
#             for system, result in all_results.items():
#                 if result and "data" in result:
#                     combined_results[system] = result["data"]
#
#             # Construct the prompt using the combined results
#             prompt = self._construct_prompt(combined_results)
#
#             # If a specific query is provided, append it to the prompt
#             if query:
#                 prompt += f"\n\nAdditional Query: {query}"
#
#             # Generate the response using the AI model
#             response = self.client.messages.create(
#                 model=self.model_name,
#                 max_tokens=1000,
#                 messages=[{"role": "user", "content": prompt}],
#             )
#             return response.content[0].text
#         except Exception as e:
#             print(f"Error in quantum interpretation: {str(e)}")
#             return f"Error analyzing quantum data: {str(e)}"
#
#     def _construct_prompt(self, results):
#         try:
#             metrics = []
#
#             for system, data in results.items():
#                 metrics.append(f"\n{system.upper()} Analysis:")
#
#                 # Add system-specific metrics
#                 if system == "quantum":
#                     frequencies = data.get("quantum_frequencies", [])
#                     metrics.append(f"Frequencies analyzed: {len(frequencies)}")
#                     metrics.append(
#                         f"State purity: {self._format_metric(data.get('purity'))}"
#                     )
#                     metrics.append(
#                         f"Quantum fidelity: {self._format_metric(data.get('fidelity'))}"
#                     )
#
#                 elif system == "qec":
#                     qec_metrics = data.get("metrics", {})
#                     metrics.append(
#                         f"Initial fidelity: {self._format_metric(qec_metrics.get('initial_fidelity'))}"
#                     )
#                     metrics.append(
#                         f"Final fidelity: {self._format_metric(qec_metrics.get('final_fidelity'))}"
#                     )
#                     metrics.append(
#                         f"QEC improvement: {self._format_metric(qec_metrics.get('improvement'), '+.1f')}%"
#                     )
#
#                 # Add more system-specific metrics as needed
#
#             # Construct final prompt
#             prompt_parts = [
#                 "Analyze the following quantum system measurements:",
#                 "",
#                 *metrics,
#                 "",
#                 "Please provide:",
#                 "1. A technical analysis of the quantum state quality",
#                 "2. Interpretation of any quantum error correction results",
#                 "3. Analysis of harmonic relationships if musical data is present",
#                 "4. Recommendations for potential improvements",
#                 "",
#                 "Explain in clear, scientific terms suitable for a physics graduate student.",
#             ]
#
#             return "\n".join(prompt_parts)
#
#         except Exception as e:
#             print(f"Error constructing prompt: {str(e)}")
#             return "Error occurred while preparing quantum analysis prompt."


import os
import numpy as np
from anthropic import Anthropic
from data.backend_data_management.data_manager import QuantumDataManager
from core.calculations.quantum.quantum_harmonic_analysis import QuantumHarmonicsAnalyzer


class QuantumInterpreter:
    def __init__(self, model_name="claude-3-opus-20240229"):
        self.model_name = model_name
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("Anthropic API key not found in environment variables")
        self.data_manager = QuantumDataManager()
        self.analyzer = QuantumHarmonicsAnalyzer()

    def _format_metric(self, value, default="N/A"):
        """Safely format a metric value"""
        if isinstance(value, (float, int)):
            return f"{value:.4f}"
        return str(default)

    def generate_summary(self, results):
        """Generate a concise summary of quantum results"""
        try:
            summary = []

            if results:
                summary.append("Comprehensive Quantum Systems Analysis:")

                # Dynamically check active systems
                active_sources = [system for system, data in results.items() if data]
                if active_sources:
                    summary.append(
                        f"• Active Data Sources: {', '.join(active_sources)}"
                    )
                else:
                    summary.append("• No active data sources found")

                # Calculate overall coherence (example for purity)
                try:
                    purities = [
                        data.get("purity")
                        for system, data in results.items()
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

    def interpret_quantum_data(self, query=None):
        try:
            # Fetch the latest results from the data manager
            all_results = self.data_manager.get_latest_results()

            # Combine all results into a single dictionary for analysis
            combined_results = {}
            for system, result in all_results.items():
                if result and "data" in result:
                    combined_results[system] = result["data"]

            # Construct the prompt using the combined results
            prompt = self._construct_prompt(combined_results)

            # If a specific query is provided, append it to the prompt
            if query:
                prompt += f"\n\nAdditional Query: {query}"

            # Generate the response using the AI model
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error in quantum interpretation: {str(e)}")
            return f"Error analyzing quantum data: {str(e)}"

    def _construct_prompt(self, results):
        """Construct a detailed prompt from results"""
        try:
            metrics = []

            for system, data in results.items():
                metrics.append(f"\n{system.upper()} Analysis:")

                # Add system-specific metrics
                if system == "quantum":
                    frequencies = data.get("quantum_frequencies", [])
                    metrics.append(f"Frequencies analyzed: {len(frequencies)}")
                    metrics.append(
                        f"State purity: {self._format_metric(data.get('purity'))}"
                    )
                    metrics.append(
                        f"Quantum fidelity: {self._format_metric(data.get('fidelity'))}"
                    )

                elif system == "qec":
                    qec_metrics = data.get("metrics", {})
                    metrics.append(
                        f"Initial fidelity: {self._format_metric(qec_metrics.get('initial_fidelity'))}"
                    )
                    metrics.append(
                        f"Final fidelity: {self._format_metric(qec_metrics.get('final_fidelity'))}"
                    )
                    metrics.append(
                        f"QEC improvement: {self._format_metric(qec_metrics.get('improvement'), '+.1f')}%"
                    )

                elif system == "melody":
                    musical_systems = data.get("musical_systems", {})
                    for sys_name, sys_data in musical_systems.items():
                        metrics.append(
                            f"{sys_name}: {len(sys_data.get('notes', []))} notes, {len(sys_data.get('frequencies', []))} frequencies"
                        )

                elif system == "fluid":
                    orig_freq = data.get("original_frequencies", [])
                    fib_freq = data.get("fibonacci_frequencies", [])
                    metrics.append(f"Original frequencies: {len(orig_freq)}")
                    metrics.append(f"Fibonacci frequencies: {len(fib_freq)}")
                    if orig_freq:
                        metrics.append(
                            f"Frequency range: {min(orig_freq):.1f} to {max(orig_freq):.1f} Hz"
                        )

                elif system == "particle":
                    positions = data.get("positions", [])
                    metrics.append(f"Number of particles: {len(positions)}")

            # Construct final prompt
            prompt_parts = [
                "Analyze the following quantum system measurements:",
                "",
                *metrics,
                "",
                "Please provide:",
                "1. A technical analysis of the quantum state quality",
                "2. Interpretation of any quantum error correction results",
                "3. Analysis of harmonic relationships if musical data is present",
                "4. Recommendations for potential improvements",
                "",
                "Explain in clear, scientific terms suitable for a physics graduate student.",
            ]

            return "\n".join(prompt_parts)

        except Exception as e:
            print(f"Error constructing prompt: {str(e)}")
            return "Error occurred while preparing quantum analysis prompt."
