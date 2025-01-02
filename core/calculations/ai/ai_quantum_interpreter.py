# ai_quantum_interpreter.py,
# developed to use advanced AI models for
# interpreting and analyzing quantum computing data.
# It integrates various functionalities for managing quantum data,
# analyzing quantum states, and providing insights into quantum measurements and operations.

import os
import numpy as np
from anthropic import Anthropic
from datetime import datetime
from storage.data_manager import QuantumDataManager
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

    def interpret_quantum_data(self, quantum_results):
        try:
            prompt = self._construct_prompt(quantum_results)
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
        try:
            # Handle case where results is just a prompt string
            if isinstance(results, dict) and "prompt" in results:
                return results["prompt"]

            metrics = []

            # Handle frequencies
            frequencies = results.get("frequencies", [])
            if frequencies:
                metrics.append(f"Frequencies analyzed: {len(frequencies)}")
                if isinstance(frequencies, (list, np.ndarray)) and len(frequencies) > 0:
                    metrics.append(
                        f"Frequency range: {min(frequencies):.2f} Hz to {max(frequencies):.2f} Hz"
                    )
                    metrics.append(f"Mean frequency: {np.mean(frequencies):.2f} Hz")

            # Handle purity/fidelity with safe formatting
            purity = results.get("purity")
            metrics.append(f"State purity: {self._format_metric(purity)}")

            fidelity = results.get("fidelity")
            metrics.append(f"Quantum fidelity: {self._format_metric(fidelity)}")

            # Handle QEC metrics with safe formatting
            qec_metrics = results.get("metrics", {})
            if qec_metrics:
                initial_fid = qec_metrics.get("initial_fidelity")
                final_fid = qec_metrics.get("final_fidelity")
                improvement = qec_metrics.get("improvement")

                metrics.append(f"Initial fidelity: {self._format_metric(initial_fid)}")
                metrics.append(f"Final fidelity: {self._format_metric(final_fid)}")
                if isinstance(improvement, (float, int)):
                    metrics.append(f"QEC improvement: {improvement:+.1f}%")
                else:
                    metrics.append("QEC improvement: N/A")

            # Handle musical systems safely
            musical_systems = results.get("musical_systems", {})
            if musical_systems:
                metrics.append("\nMusical System Analysis:")
                for system, data in musical_systems.items():
                    if isinstance(data, dict) and "notes" in data:
                        metrics.append(
                            f"{system}: {', '.join(map(str, data['notes']))}"
                        )

            # Get stored data from data manager
            stored_results = self.data_manager.get_all_results()
            if stored_results:
                metrics.append("\nAvailable Analysis Types:")
                for key, value in stored_results.items():
                    if value is not None:
                        metrics.append(f"- {key.capitalize()} analysis data")

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
