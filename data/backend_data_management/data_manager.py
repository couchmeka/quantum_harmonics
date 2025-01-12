import traceback
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from data.backend_data_management.data_validator import DataValidator


class QuantumDataManager:
    def __init__(self):
        # Initialize storage for each tab
        self.unified_system = []
        self.data_manager = []
        self.quantum_results = []  # From QuantumAnalysisTab
        self.qec_results = []  # From QECAnalysisTab
        self.melody_results = []  # From MelodyAnalysisTab
        self.circuit_results = []  # From CircuitTab
        self.fluid_results = []  # From FluidDynamicsTab
        self.particle_results = []  # From ParticleSimulationTab
        self.last_update_time = None
        self._last_qec_update = None

        # Constants for harmonic analysis
        self.PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.fibonacci_seq = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        self.fibonacci_freqs = [f * 100 for f in self.fibonacci_seq]
        self.base_freq = 256  # Reference frequency

    def update_quantum_results(self, results):
        try:
            validated_data = DataValidator.validate_quantum_data(results)
            if validated_data:
                timestamp = datetime.now()
                # Ensure frequencies are appended correctly
                frequencies = validated_data.get("quantum_frequencies", [])
                if isinstance(frequencies, list) and len(frequencies) > 0:
                    validated_data["quantum_frequencies"] = frequencies
                else:
                    print("Warning: No valid frequencies found!")

                self.quantum_results.append(
                    {"timestamp": timestamp, "data": validated_data}
                )
                self.last_update_time = timestamp
        except Exception as e:
            print(f"Error in update_quantum_results: {str(e)}")

    def update_qec_results(self, results):
        """Update QEC analysis results with validation"""
        try:
            if not hasattr(self, "_last_qec_update"):
                self._last_qec_update = None

            # Check if this is a duplicate update
            current_time = datetime.now()
            if (
                self._last_qec_update is not None
                and (current_time - self._last_qec_update).total_seconds() < 1
            ):
                print("Skipping duplicate QEC update")
                return

            validated_data = DataValidator.validate_qec_data(results)
            if validated_data is not None:
                validated_data["system_name"] = "qec"
                self.qec_results.append(
                    {"timestamp": current_time, "data": validated_data}
                )
                self._last_qec_update = current_time
                print("Successfully updated QEC results")
        except Exception as e:
            print(f"Error in update_qec_results: {str(e)}")
            traceback.print_exc()

    def update_melody_results(self, results):
        """Update melody analysis results"""
        if results:
            timestamp = datetime.now()
            results["system_name"] = "melody"  # Add system name here
            self.melody_results.append({"timestamp": timestamp, "data": results})
            self.last_update_time = timestamp

    def update_fluid_results(self, results):
        """
        Update fluid dynamics results with validation.
        """
        try:
            validated_data = DataValidator.validate_fluid_data(results)
            if validated_data:
                timestamp = datetime.now()
                validated_data["system_name"] = "fluid"
                self.fluid_results.append(
                    {"timestamp": timestamp, "data": validated_data}
                )
                self.last_update_time = timestamp
                print(f"Fluid results successfully updated: {validated_data}")
            else:
                print("Fluid results update failed: Validation error.")
        except Exception as e:
            print(f"Error in update_fluid_results: {str(e)}")
            traceback.print_exc()

    def update_particle_results(self, results):
        """Update particle simulation results with validation"""
        try:
            validated_data = DataValidator.validate_particle_data(results)
            if validated_data is not None:
                timestamp = datetime.now()
                validated_data["system_name"] = "particle"  # Add system name here
                self.particle_results.append(
                    {"timestamp": timestamp, "data": validated_data}
                )
                self.last_update_time = timestamp
            else:
                print("Warning: Particle data validation failed")
        except Exception as e:
            print(f"Error in update_particle_results: {str(e)}")
            traceback.print_exc()

    def collect_tab_data(self):
        """Collect and aggregate data from all tabs"""
        aggregated_data = {
            "quantum": {
                "frequencies": [],
                "statevector": None,
                "fibonacci_analysis": None,
            }
        }

        # Collect from QuantumAnalysisTab with validated data
        if self.quantum_results:
            latest = self.quantum_results[-1]["data"]
            aggregated_data["quantum"].update(
                {
                    "frequencies": latest.get("quantum_frequencies", []),
                    "statevector": latest.get("statevector", None),
                    "purity": latest.get("purity", 0),
                    "fidelity": latest.get("fidelity", 0),
                    "fibonacci_analysis": latest.get("fibonacci_analysis", {}),
                }
            )

        # Collect from other tabs...
        if self.melody_results:
            latest = self.melody_results[-1]["data"]
            aggregated_data["melody"] = {
                "notes": latest.get("notes", []),
                "frequencies": latest.get("frequencies", []),
                "musical_systems": latest.get("musical_systems", {}),
            }

        if self.fluid_results:
            latest = self.fluid_results[-1]["data"]
            aggregated_data["fluid"] = {
                "original_frequencies": latest.get("original_frequencies", []),
                "fibonacci_frequencies": latest.get("fibonacci_frequencies", []),
            }

        if self.qec_results:
            latest = self.qec_results[-1]["data"]
            aggregated_data["qec"] = {
                "metrics": latest.get("metrics", {}),
                "qec_type": latest.get("qec_type", ""),
                "material": latest.get("material", {}),
                "fidelity": latest.get("fidelity", 0),
            }

        if self.particle_results:
            latest = self.particle_results[-1]["data"]
            aggregated_data["particle"] = {
                "positions": latest.get("positions", []),
                "velocities": latest.get("velocities", []),
                "mode": latest.get("mode", ""),
            }

        return aggregated_data

    def get_all_frequencies(self):
        """Get all frequencies from all systems"""
        frequencies = {"quantum": [], "melody": [], "fluid": [], "fibonacci": []}

        # Collect from quantum results
        for result in self.quantum_results:
            if "data" in result and "quantum_frequencies" in result["data"]:
                frequencies["quantum"].extend(result["data"]["quantum_frequencies"])

        # Collect from melody results
        for result in self.melody_results:
            if "data" in result and "frequencies" in result["data"]:
                frequencies["melody"].extend(result["data"]["frequencies"])

        # Collect from fluid results
        for result in self.fluid_results:
            if "data" in result:
                if "original_frequencies" in result["data"]:
                    frequencies["fluid"].extend(result["data"]["original_frequencies"])
                if "fibonacci_frequencies" in result["data"]:
                    frequencies["fibonacci"].extend(
                        result["data"]["fibonacci_frequencies"]
                    )

        return frequencies

    def get_all_results(self):
        """
        Retrieve all results with proper validation and system names.
        Returns a dictionary of validated results for each system.
        """
        try:
            results = {}

            # Process Quantum Results
            if self.quantum_results:
                validated_quantum = []
                for result in self.quantum_results:
                    if isinstance(result, dict) and "data" in result:
                        validated_data = DataValidator.validate_quantum_data(
                            result["data"]
                        )
                        if validated_data:
                            validated_quantum.append(
                                {
                                    "timestamp": result.get("timestamp"),
                                    "data": {
                                        **validated_data,
                                        "system_name": "quantum",
                                    },
                                }
                            )
                if validated_quantum:
                    results["quantum"] = validated_quantum

            # Process QEC Results
            if self.qec_results:
                validated_qec = []
                for result in self.qec_results:
                    if isinstance(result, dict) and "data" in result:
                        validated_data = DataValidator.validate_qec_data(result["data"])
                        if validated_data:
                            validated_qec.append(
                                {
                                    "timestamp": result.get("timestamp"),
                                    "data": {**validated_data, "system_name": "qec"},
                                }
                            )
                if validated_qec:
                    results["qec"] = validated_qec

            # Process Melody Results
            if self.melody_results:
                validated_melody = []
                for result in self.melody_results:
                    if isinstance(result, dict) and "data" in result:
                        # Assume melody data is already validated during update
                        melody_data = result["data"]
                        if melody_data:
                            validated_melody.append(
                                {
                                    "timestamp": result.get("timestamp"),
                                    "data": {**melody_data, "system_name": "melody"},
                                }
                            )
                if validated_melody:
                    results["melody"] = validated_melody

            # Process Fluid Results
            if self.fluid_results:
                validated_fluid = []
                for result in self.fluid_results:
                    if isinstance(result, dict) and "data" in result:
                        fluid_data = result["data"]
                        if fluid_data:
                            validated_fluid.append(
                                {
                                    "timestamp": result.get("timestamp"),
                                    "data": {**fluid_data, "system_name": "fluid"},
                                }
                            )
                if validated_fluid:
                    results["fluid"] = validated_fluid

            # Process Particle Results
            if self.particle_results:
                validated_particle = []
                for result in self.particle_results:
                    if isinstance(result, dict) and "data" in result:
                        validated_data = DataValidator.validate_particle_data(
                            result["data"]
                        )
                        if validated_data:
                            validated_particle.append(
                                {
                                    "timestamp": result.get("timestamp"),
                                    "data": {
                                        **validated_data,
                                        "system_name": "particle",
                                    },
                                }
                            )
                if validated_particle:
                    results["particle"] = validated_particle
                print(results)

            return results

        except Exception as e:
            print(f"Error in get_all_results: {str(e)}")
            traceback.print_exc()
            return {}

    def get_latest_results(self):
        """Get the most recent valid result for each system."""
        try:
            latest_results = {}

            # Quantum Results
            if self.quantum_results and isinstance(self.quantum_results, list):
                latest = self.quantum_results[-1]
                if isinstance(latest, dict) and "data" in latest:
                    validated_data = DataValidator.validate_quantum_data(latest["data"])
                    if validated_data:
                        latest_results["quantum"] = {
                            "timestamp": latest.get("timestamp"),
                            "data": {**validated_data, "system_name": "quantum"},
                        }

            # QEC Results
            if self.qec_results and isinstance(self.qec_results, list):
                latest = self.qec_results[-1]
                if isinstance(latest, dict) and "data" in latest:
                    validated_data = DataValidator.validate_qec_data(latest["data"])
                    if validated_data:
                        latest_results["qec"] = {
                            "timestamp": latest.get("timestamp"),
                            "data": {**validated_data, "system_name": "qec"},
                        }

            # Melody Results
            if self.melody_results and isinstance(self.melody_results, list):
                latest = self.melody_results[-1]
                if isinstance(latest, dict) and "data" in latest:
                    latest_results["melody"] = {
                        "timestamp": latest.get("timestamp"),
                        "data": {**latest["data"], "system_name": "melody"},
                    }

            # Fluid Results
            if self.fluid_results and isinstance(self.fluid_results, list):
                latest = self.fluid_results[-1]
                if isinstance(latest, dict) and "data" in latest:
                    latest_results["fluid"] = {
                        "timestamp": latest.get("timestamp"),
                        "data": {**latest["data"], "system_name": "fluid"},
                    }

            # Particle Results
            if self.particle_results and isinstance(self.particle_results, list):
                latest = self.particle_results[-1]
                if isinstance(latest, dict) and "data" in latest:
                    validated_data = DataValidator.validate_particle_data(
                        latest["data"]
                    )
                    if validated_data:
                        latest_results["particle"] = {
                            "timestamp": latest.get("timestamp"),
                            "data": {**validated_data, "system_name": "particle"},
                        }

            return latest_results

        except Exception as e:
            print(f"Error in get_latest_results: {str(e)}")
            traceback.print_exc()
            return {}

    # Helper method to safely get the latest result
    def _get_latest_valid_result(
        self, results_list, validator_func=None, system_name=None
    ):
        """Helper method to safely get the latest valid result for a system."""
        try:
            if not results_list or not isinstance(results_list, list):
                return None

            latest = results_list[-1]
            if not isinstance(latest, dict) or "data" not in latest:
                return None

            data = latest["data"]
            if validator_func:
                data = validator_func(data)
                if not data:
                    return None

            if system_name:
                data["system_name"] = system_name

            return {"timestamp": latest.get("timestamp"), "data": data}
        except Exception as e:
            print(f"Error getting latest result for {system_name}: {str(e)}")
            return None

    def get_filtered_results(self, system_name):
        """
        Get filtered results for a specific system with validation.

        Args:
            system_name (str): Name of the system to filter for ('quantum', 'qec', etc.)
        """
        try:
            all_results = self.get_all_results()
            return (
                {system_name: all_results.get(system_name, [])}
                if system_name in all_results
                else {}
            )

        except Exception as e:
            print(f"Error in get_filtered_results: {str(e)}")
            return {}

    def get_latest_quantum_results(self):
        """Get the most recent quantum analysis results"""
        if self.quantum_results:
            return self.quantum_results[-1]["data"]
        return None

    def analyze_ml_patterns(self):
        try:
            feature_rows = []
            all_results = self.get_all_results()

            # Extract features for each system
            for system_name, results_list in all_results.items():
                if not isinstance(results_list, list):
                    continue

                for result in results_list:
                    if not result or "data" not in result:
                        continue

                    result_data = result["data"]
                    features = {"system_name": system_name}

                    # Handle numerical data only
                    for key, values in result_data.items():
                        if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                            try:
                                # Convert to numpy array and ensure numeric type
                                values_arr = np.array(values, dtype=np.float64)
                                if values_arr.size > 0:
                                    features[f"{key}_mean"] = np.mean(values_arr)
                                    features[f"{key}_std"] = np.std(values_arr)
                            except (ValueError, TypeError):
                                print(f"Skipping non-numeric values in {key}")
                                continue
                        elif isinstance(values, (int, float)):
                            features[key] = float(values)

                    if features:
                        feature_rows.append(features)

            if len(feature_rows) < 2:
                print("Insufficient data for ML analysis")
                return None

            # Create DataFrame with only numeric columns
            df = pd.DataFrame(feature_rows)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                print("Insufficient numeric features for analysis")
                return None

            numeric_cols = df.select_dtypes(include=["number"]).columns
            # Clean the data
            X = df[numeric_cols].fillna(0)  # Replace NaN with 0
            X = X.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
            X = X.fillna(0)  # Replace remaining NaNs with 0

            # Optional: Clip large values
            X = np.clip(X, -1e10, 1e10)

            # Apply StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # PCA Analysis
            n_components = min(3, X_scaled.shape[1])
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)

            # Clustering
            n_clusters = min(max(2, len(X_scaled) // 2), 5)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)

            # Anomaly Detection
            iso_forest = IsolationForest(random_state=42)
            anomalies = iso_forest.fit_predict(X_scaled)

            # Create feature importance dict with proper numeric handling
            feature_importance = {}
            for i, col in enumerate(numeric_cols):
                if i < len(pca.components_[0]):
                    feature_importance[col] = float(abs(pca.components_[0][i]))

            return {
                "pca_components": X_pca.tolist(),
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "clusters": clusters.tolist(),
                "anomalies": anomalies.tolist(),
                "feature_importance": feature_importance,
                "n_samples": len(X),
                "n_features": len(numeric_cols),
            }

        except Exception as e:
            print(f"Error in analyze_ml_patterns: {str(e)}")
            import traceback

            traceback.print_exc()
            return None
