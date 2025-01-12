import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from datetime import datetime
from typing import Dict, List


class UnifiedAIAnalyzer:
    def __init__(self):
        # Constants for analysis
        self.PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.fibonacci_seq = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        self.fibonacci_freqs = [f * 100 for f in self.fibonacci_seq]
        self.pythagorean_ratios = [
            1.0,
            9 / 8,
            81 / 64,
            4 / 3,
            3 / 2,
            27 / 16,
            243 / 128,
            2.0,
        ]
        self.base_freq = 256  # Reference frequency for Pythagorean tuning

    def analyze_unified_system(
            self, quantum_data: Dict, melody_data: Dict, fluid_data: Dict, qec_data: Dict
    ) -> Dict:
        """
        Perform unified analysis across all systems
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "quantum_analysis": {},
            "harmonic_analysis": {},
            "correlation_analysis": {},
            "pattern_analysis": {},
            "metrics": {},
        }

        try:
            # 1. Quantum State Analysis
            quantum_results = self._analyze_quantum_state(quantum_data)
            results["quantum_analysis"] = quantum_results

            # 2. Harmonic Pattern Analysis
            harmonic_results = self._analyze_harmonic_patterns(
                quantum_data=quantum_data,
                melody_data=melody_data,
                fluid_data=fluid_data,
            )
            results["harmonic_analysis"] = harmonic_results

            # 3. Cross-System Correlation Analysis
            correlation_results = self._analyze_cross_system_correlations(
                quantum_data=quantum_data,
                melody_data=melody_data,
                fluid_data=fluid_data,
                qec_data=qec_data,
            )
            results["correlation_analysis"] = correlation_results

            # 4. Pattern Recognition
            pattern_results = self._analyze_patterns(
                quantum_data=quantum_data,
                melody_data=melody_data,
                fluid_data=fluid_data,
                qec_data=qec_data,
            )
            results["pattern_analysis"] = pattern_results

            # 5. Calculate Unified Metrics
            metrics = self._calculate_unified_metrics(
                quantum_results=quantum_results,
                harmonic_results=harmonic_results,
                correlation_results=correlation_results,
                pattern_results=pattern_results,
            )
            results["metrics"] = metrics

            return results

        except Exception as e:
            print(f"Error in unified analysis: {str(e)}")
            return results

    def _analyze_quantum_state(self, quantum_data: Dict) -> Dict:
        """
        Analyze quantum state properties and stability
        """
        results = {}

        try:
            # Extract frequencies and validate
            frequencies = quantum_data.get("quantum_frequencies", [])
            if not frequencies:
                return results

            # Calculate quantum metrics
            if "statevector" in quantum_data:
                statevector = np.array(quantum_data["statevector"])
                density_matrix = np.outer(statevector, statevector.conj())

                # Calculate core quantum metrics
                purity = np.real(np.trace(np.matmul(density_matrix, density_matrix)))
                fidelity = np.max(np.abs(statevector) ** 2)

                # Calculate coherence with harmonic weighting
                fib_contribution = self._analyze_fibonacci_contribution(frequencies)
                pyth_contribution = self._analyze_pythagorean_contribution(frequencies)

                off_diagonal_sum = np.sum(
                    np.abs(density_matrix - np.diag(np.diag(density_matrix)))
                )
                coherence = (
                        off_diagonal_sum * (1 + fib_contribution + pyth_contribution) / 3
                )

                results.update(
                    {
                        "purity": float(purity),
                        "fidelity": float(fidelity),
                        "coherence": float(coherence),
                        "harmonic_contributions": {
                            "fibonacci": float(fib_contribution),
                            "pythagorean": float(pyth_contribution),
                        },
                    }
                )

            # Analyze frequency distribution
            if len(frequencies) > 0:
                freq_array = np.array(frequencies)
                results.update(
                    {
                        "frequency_stats": {
                            "mean": float(np.mean(freq_array)),
                            "std": float(np.std(freq_array)),
                            "min": float(np.min(freq_array)),
                            "max": float(np.max(freq_array)),
                        }
                    }
                )

        except Exception as e:
            print(f"Error in quantum state analysis: {str(e)}")

        return results

    def _analyze_harmonic_patterns(self, **system_data: Dict) -> Dict:
        """
        Analyze harmonic patterns across quantum, melodic, and fluid systems
        """
        results = {
            "fibonacci_patterns": {},
            "pythagorean_patterns": {},
            "cross_system_harmonics": {},
        }

        try:
            # Collect all frequencies
            all_frequencies = []
            frequency_sources = []

            # Extract frequencies from each system
            if "quantum_data" in system_data:
                quantum_freqs = system_data["quantum_data"].get(
                    "quantum_frequencies", []
                )
                if quantum_freqs:
                    all_frequencies.extend(quantum_freqs)
                    frequency_sources.extend(["quantum"] * len(quantum_freqs))

            if "melody_data" in system_data:
                melody_freqs = system_data["melody_data"].get("frequencies", [])
                if melody_freqs:
                    all_frequencies.extend(melody_freqs)
                    frequency_sources.extend(["melody"] * len(melody_freqs))

            if "fluid_data" in system_data:
                fluid_freqs = system_data["fluid_data"].get("original_frequencies", [])
                if fluid_freqs:
                    all_frequencies.extend(fluid_freqs)
                    frequency_sources.extend(["fluid"] * len(fluid_freqs))

            if not all_frequencies:
                return results

            # Analyze Fibonacci patterns
            fib_patterns = self._analyze_fibonacci_contribution(all_frequencies)
            results["fibonacci_patterns"] = {
                "contribution": float(fib_patterns),
                "matched_ratios": self._find_fibonacci_matches(all_frequencies),
            }

            # Analyze Pythagorean patterns
            pyth_patterns = self._analyze_pythagorean_contribution(all_frequencies)
            results["pythagorean_patterns"] = {
                "contribution": float(pyth_patterns),
                "matched_ratios": self._find_pythagorean_matches(all_frequencies),
            }

            # Analyze cross-system harmonics
            if len(all_frequencies) > 1:
                cross_harmonics = self._analyze_cross_system_harmonics(
                    all_frequencies, frequency_sources
                )
                results["cross_system_harmonics"] = cross_harmonics

        except Exception as e:
            print(f"Error in harmonic pattern analysis: {str(e)}")

        return results

    def _analyze_cross_system_correlations(self, **system_data: Dict) -> Dict:
        """
        Analyze correlations between different systems
        """
        results = {
            "quantum_melody_correlation": None,
            "quantum_fluid_correlation": None,
            "melody_fluid_correlation": None,
            "qec_impact": None,
        }

        try:
            # Extract frequency data from each system
            quantum_freqs = np.array(
                system_data.get("quantum_data", {}).get("quantum_frequencies", [])
            )
            melody_freqs = np.array(
                system_data.get("melody_data", {}).get("frequencies", [])
            )
            fluid_freqs = np.array(
                system_data.get("fluid_data", {}).get("original_frequencies", [])
            )

            # Calculate correlations between systems
            if len(quantum_freqs) > 0 and len(melody_freqs) > 0:
                # Match lengths for correlation
                min_len = min(len(quantum_freqs), len(melody_freqs))
                correlation = np.corrcoef(
                    quantum_freqs[:min_len], melody_freqs[:min_len]
                )[0, 1]
                results["quantum_melody_correlation"] = float(correlation)

            if len(quantum_freqs) > 0 and len(fluid_freqs) > 0:
                min_len = min(len(quantum_freqs), len(fluid_freqs))
                correlation = np.corrcoef(
                    quantum_freqs[:min_len], fluid_freqs[:min_len]
                )[0, 1]
                results["quantum_fluid_correlation"] = float(correlation)

            if len(melody_freqs) > 0 and len(fluid_freqs) > 0:
                min_len = min(len(melody_freqs), len(fluid_freqs))
                correlation = np.corrcoef(
                    melody_freqs[:min_len], fluid_freqs[:min_len]
                )[0, 1]
                results["melody_fluid_correlation"] = float(correlation)

            # Analyze QEC impact if present
            if "qec_data" in system_data and system_data["qec_data"]:
                qec_metrics = system_data["qec_data"].get("metrics", {})
                if qec_metrics:
                    results["qec_impact"] = self._analyze_qec_impact(qec_metrics)

        except Exception as e:
            print(f"Error in cross-system correlation analysis: {str(e)}")

        return results

    def _analyze_patterns(self, **system_data: Dict) -> Dict:
        """
        Perform pattern recognition across all systems
        """
        results = {"clusters": None, "anomalies": None, "feature_importance": None}

        try:
            # Extract features from all systems
            features = self._extract_unified_features(system_data)

            if not features or len(features) < 2:
                return results

            # Convert to numpy array for analysis
            X = np.array(features)

            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform PCA
            pca = PCA(n_components=min(3, X_scaled.shape[1]))
            X_pca = pca.fit_transform(X_scaled)

            # Clustering
            n_clusters = min(max(2, len(X_scaled) // 2), 5)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)

            # Anomaly detection
            isolation_forest = IsolationForest(random_state=42)
            anomalies = isolation_forest.fit_predict(X_scaled)

            results.update(
                {
                    "clusters": clusters.tolist(),
                    "cluster_centers": kmeans.cluster_centers_.tolist(),
                    "anomalies": anomalies.tolist(),
                    "pca_components": X_pca.tolist(),
                    "explained_variance": pca.explained_variance_ratio_.tolist(),
                    "feature_importance": dict(
                        zip(
                            ["feature_" + str(i) for i in range(X.shape[1])],
                            np.abs(pca.components_[0]),
                        )
                    ),
                }
            )

        except Exception as e:
            print(f"Error in pattern analysis: {str(e)}")

        return results

    def _calculate_unified_metrics(self, **analysis_results: Dict) -> Dict:
        """
        Calculate unified metrics across all analyses
        """
        metrics = {
            "system_coherence": None,
            "harmonic_stability": None,
            "cross_system_coupling": None,
            "pattern_strength": None,
        }

        try:
            # Calculate system coherence
            quantum_results = analysis_results.get("quantum_results", {})
            if "coherence" in quantum_results:
                metrics["system_coherence"] = quantum_results["coherence"]

            # Calculate harmonic stability
            harmonic_results = analysis_results.get("harmonic_results", {})
            if "fibonacci_patterns" in harmonic_results:
                fib_contrib = harmonic_results["fibonacci_patterns"].get(
                    "contribution", 0
                )
                pyth_contrib = harmonic_results["pythagorean_patterns"].get(
                    "contribution", 0
                )
                metrics["harmonic_stability"] = (fib_contrib + pyth_contrib) / 2

            # Calculate cross-system coupling
            correlation_results = analysis_results.get("correlation_results", {})
            correlations = [
                corr
                for corr in correlation_results.values()
                if isinstance(corr, (int, float)) and corr is not None
            ]
            if correlations:
                metrics["cross_system_coupling"] = np.mean(correlations)

            # Calculate pattern strength
            pattern_results = analysis_results.get("pattern_results", {})
            if "explained_variance" in pattern_results:
                metrics["pattern_strength"] = np.mean(
                    pattern_results["explained_variance"]
                )

        except Exception as e:
            print(f"Error calculating unified metrics: {str(e)}")

        return metrics

    # Helper methods
    def _analyze_fibonacci_contribution(self, frequencies: List[float]) -> float:
        """Calculate Fibonacci sequence contribution to frequencies"""
        if not frequencies:
            return 0.0

        contribution = 0
        for f in frequencies:
            closest_fib = min(self.fibonacci_freqs, key=lambda x: abs(x - f))
            contribution += 1.0 / (1.0 + abs(f - closest_fib))
        return contribution / len(frequencies)

    def _analyze_pythagorean_contribution(self, frequencies: List[float]) -> float:
        """Calculate Pythagorean tuning contribution to frequencies"""
        if not frequencies:
            return 0.0

        contribution = 0
        pythagorean_freqs = [
            self.base_freq * ratio for ratio in self.pythagorean_ratios
        ]

        for f in frequencies:
            closest_pyth = min(pythagorean_freqs, key=lambda x: abs(x - f))
            contribution += 1.0 / (1.0 + abs(f - closest_pyth))
        return contribution / len(frequencies)

    def _find_fibonacci_matches(self, frequencies: List[float]) -> List[Dict]:
        """Find matches between frequencies and Fibonacci-derived frequencies"""
        matches = []
        for freq in frequencies:
            closest_fib = min(self.fibonacci_freqs, key=lambda x: abs(x - freq))
            if closest_fib != 0:
                ratio = freq / closest_fib
                deviation = abs(freq - closest_fib) / closest_fib
                if deviation < 0.1:  # 10% tolerance
                    matches.append(
                        {
                            "frequency": float(freq),
                            "fibonacci_freq": float(closest_fib),
                            "ratio": float(ratio),
                            "deviation": float(deviation),
                        }
                    )
        return matches

    def _find_pythagorean_matches(self, frequencies: List[float]) -> List[Dict]:
        """Find matches between frequencies and Pythagorean-tuned frequencies"""
        matches = []
        pythagorean_freqs = [
            self.base_freq * ratio for ratio in self.pythagorean_ratios
        ]

        for freq in frequencies:
            closest_pyth = min(pythagorean_freqs, key=lambda x: abs(x - freq))
            if closest_pyth != 0:
                ratio = freq / closest_pyth
                deviation = abs(freq - closest_pyth) / closest_pyth
                if deviation < 0.1:  # 10% tolerance
                    matches.append(
                        {
                            "frequency": float(freq),
                            "pythagorean_freq": float(closest_pyth),
                            "ratio": float(ratio),
                            "deviation": float(deviation),
                        }
                    )
        return matches

    def _analyze_cross_system_harmonics(
            self, frequencies: List[float], sources: List[str]
    ) -> Dict:
        """Analyze harmonic relationships between different systems"""
        results = {"harmonic_ratios": [], "system_interactions": {}}

        # Calculate ratios between frequencies from different systems
        for i, (freq1, source1) in enumerate(zip(frequencies, sources)):
            for j, (freq2, source2) in enumerate(
                    zip(frequencies[i + 1:], sources[i + 1:])
            ):
                if freq2 != 0:
                    ratio = freq1 / freq2
                    interaction_key = f"{source1}_{source2}"

                    # Store the ratio and its source systems
                    ratio_info = {
                        "ratio": float(ratio),
                        "freq1": float(freq1),
                        "freq2": float(freq2),
                        "source1": source1,
                        "source2": source2,
                    }
                    results["harmonic_ratios"].append(ratio_info)

                    # Track system interactions
                    if interaction_key not in results["system_interactions"]:
                        results["system_interactions"][interaction_key] = []
                    results["system_interactions"][interaction_key].append(ratio)

        # Calculate average ratios for each system interaction
        for key in results["system_interactions"]:
            ratios = results["system_interactions"][key]
            results["system_interactions"][key] = {
                "mean_ratio": float(np.mean(ratios)),
                "std_ratio": float(np.std(ratios)),
                "count": len(ratios),
            }

        return results

    def _extract_unified_features(self, system_data: Dict) -> np.ndarray:
        """Extract unified features from all systems for pattern analysis"""
        features = []

        # Process each result
        for name, data in system_data.items():
            if not data:
                continue

            feature_vector = []

            # Extract frequencies
            frequencies = []
            if name == "quantum_data":
                frequencies = data.get("quantum_frequencies", [])
            elif name == "melody_data":
                frequencies = data.get("frequencies", [])
            elif name == "fluid_data":
                frequencies = data.get("original_frequencies", [])

            if frequencies:
                # Basic statistical features
                feature_vector.extend(
                    [
                        np.mean(frequencies),
                        np.std(frequencies),
                        np.min(frequencies),
                        np.max(frequencies),
                    ]
                )

                # Harmonic features
                feature_vector.extend(
                    [
                        self._analyze_fibonacci_contribution(frequencies),
                        self._analyze_pythagorean_contribution(frequencies),
                    ]
                )

            # Add system-specific features
            if name == "quantum_data":
                feature_vector.extend([data.get("purity", 0), data.get("fidelity", 0)])
            elif name == "qec_data":
                metrics = data.get("metrics", {})
                feature_vector.extend(
                    [
                        metrics.get("initial_fidelity", 0),
                        metrics.get("final_fidelity", 0),
                    ]
                )

            if feature_vector:
                features.append(feature_vector)

        return np.array(features) if features else None

    def _analyze_qec_impact(self, qec_metrics: Dict) -> Dict:
        """Analyze the impact of quantum error correction"""
        return {
            "fidelity_improvement": float(
                qec_metrics.get("final_fidelity", 0)
                - qec_metrics.get("initial_fidelity", 0)
            ),
            "error_rate": float(qec_metrics.get("error_rate", 0)),
            "stability_score": float(
                qec_metrics.get("final_fidelity", 0)
                * (1 - qec_metrics.get("error_rate", 0))
            ),
        }
