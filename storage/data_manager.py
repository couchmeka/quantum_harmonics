# data_manager.py
import traceback
from datetime import datetime
import copy

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd


class QuantumDataManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QuantumDataManager, cls).__new__(cls)
            # Move initialization here
            cls._instance.fluid_results = []
            cls._instance.quantum_results = []
            cls._instance.qec_results = []
            cls._instance.melody_results = []
            cls._instance.circuit_results = []
            cls._instance.particle_results = []
            cls._instance.analysis_history = []
            cls._instance.last_update_time = None
        return cls._instance

    def __init__(self):
        pass

    def _initialize(self):
        # Make sure lists are initialized here too
        if self.fluid_results is None:
            self.fluid_results = []
        if self.quantum_results is None:
            self.quantum_results = []
        if self.qec_results is None:
            self.qec_results = []
        if self.melody_results is None:
            self.melody_results = []
        if self.particle_results is None:
            self.particle_results = []
        if self.circuit_results is None:
            self.circuit_results = []

    # Add to QuantumDataManager class
    def get_note_statistics(self):
        """Get statistics about note usage across runs"""
        note_stats = {"combinations": {}, "frequencies": {}, "systems": {}}

        for result in self.melody_results:
            if "musical_systems" in result["data"]:
                for system, data in result["data"]["musical_systems"].items():
                    if "notes" in data:
                        notes = tuple(sorted(data["notes"]))
                        note_stats["combinations"][notes] = (
                            note_stats["combinations"].get(notes, 0) + 1
                        )

        return note_stats

    def update_fluid_results(self, results):
        if results:  # Add check for empty results
            self.fluid_results.append(
                {"timestamp": datetime.now(), "data": copy.deepcopy(results)}
            )
            self._update_timestamp()
            print(f"Added fluid results. Total runs: {len(self.fluid_results)}")

    def update_quantum_results(self, results):
        if results:  # Add check for empty results
            self.quantum_results.append(
                {"timestamp": datetime.now(), "data": copy.deepcopy(results)}
            )
            self._update_timestamp()
            print(f"Added quantum results. Total runs: {len(self.quantum_results)}")

    def update_qec_results(self, results):
        self.qec_results.append(
            {"timestamp": datetime.now(), "data": copy.deepcopy(results)}
        )
        self._update_timestamp()
        print(f"Added QEC results. Total runs: {len(self.qec_results)}")

    def update_melody_results(self, results):
        if results:  # Add check for empty results
            self.melody_results.append(
                {"timestamp": datetime.now(), "data": copy.deepcopy(results)}
            )
            self._update_timestamp()
            print(f"Added melody results. Total runs: {len(self.melody_results)}")

    # Add to data_manager.py
    def update_particle_results(self, results):
        """Store particle simulation results with timestamp"""
        self.particle_results.append(
            {"timestamp": datetime.now(), "data": copy.deepcopy(results)}
        )
        self._update_timestamp()
        print(f"Added particle results. Total runs: {len(self.particle_results)}")

    def _update_timestamp(self):
        self.last_update_time = datetime.now()
        print(f"Updated timestamp: {self.last_update_time}")

    def get_all_results(self):
        return {
            "fluid": self.fluid_results,
            "quantum": self.quantum_results,
            "qec": self.qec_results,
            "melody": self.melody_results,
            "particle": self.particle_results,  # Add this line
        }

    def get_latest_results(self):
        latest_results = {
            "fluid": self._format_latest_result(self.fluid_results),
            "quantum": self._format_latest_result(self.quantum_results),
            "qec": self._format_latest_result(self.qec_results),
            "melody": self._format_latest_result(self.melody_results),
            "circuit": self._format_latest_result(self.circuit_results),
            "particle": self._format_latest_result(self.particle_results),
        }
        return latest_results

    def _format_latest_result(self, results):
        """Helper to format latest result with proper data extraction"""
        if results and len(results) > 0:
            latest = results[-1]
            return {
                "data": latest["data"],
                "timestamp": latest["timestamp"],
                "statevector": latest["data"].get("statevector", None),
                "frequencies": latest["data"].get("frequencies", []),
                "metrics": latest["data"].get("metrics", {}),
            }
        return None

    def has_data(self):
        has_data = any(
            [
                self.fluid_results,
                self.quantum_results,
                self.qec_results,
                self.melody_results,
                self.circuit_results,
                self.particle_results,  # Add this line
            ]
        )

    def analyze_data_patterns(self):
        try:
            # Create features dictionary
            features = {}

            # Quantum features
            if self.quantum_results:
                quantum_df = pd.DataFrame([r["data"] for r in self.quantum_results])
                features.update(
                    {
                        "frequencies": quantum_df["frequencies"].apply(
                            lambda x: np.mean(x) if isinstance(x, list) else 0
                        ),
                        "purity": quantum_df.get("purity", pd.Series([0])),
                        "fidelity": quantum_df.get("fidelity", pd.Series([0])),
                    }
                )

            # Melody features
            if self.melody_results:
                melody_df = pd.DataFrame([r["data"] for r in self.melody_results])
                features.update(
                    {
                        "notes_count": (
                            melody_df["notes"].apply(len)
                            if "notes" in melody_df
                            else pd.Series([0])
                        ),
                        "melody_frequencies": melody_df["frequencies"].apply(
                            lambda x: len(x) if isinstance(x, list) else 0
                        ),
                    }
                )

            # QEC features
            if self.qec_results:
                qec_df = pd.DataFrame([r["data"] for r in self.qec_results])
                features.update(
                    {
                        "qec_initial": qec_df["metrics"].apply(
                            lambda x: x.get("initial_fidelity", 0)
                        ),
                        "qec_final": qec_df["metrics"].apply(
                            lambda x: x.get("final_fidelity", 0)
                        ),
                    }
                )

            # Fluid features
            if self.fluid_results:
                fluid_df = pd.DataFrame([r["data"] for r in self.fluid_results])
                features.update(
                    {
                        "fluid_orig_freq": fluid_df["original_frequencies"].apply(
                            lambda x: len(x) if isinstance(x, list) else 0
                        ),
                        "fluid_fib_freq": fluid_df["fibonacci_frequencies"].apply(
                            lambda x: len(x) if isinstance(x, list) else 0
                        ),
                    }
                )

            # Particle features
            if self.particle_results:
                particle_df = pd.DataFrame([r["data"] for r in self.particle_results])
                features.update(
                    {
                        "particle_positions": particle_df["positions"].apply(
                            lambda x: len(x) if isinstance(x, list) else 0
                        ),
                        "particle_velocities": particle_df["velocities"].apply(
                            lambda x: len(x) if isinstance(x, list) else 0
                        ),
                        "particle_mode": particle_df["mode"]
                        .astype("category")
                        .cat.codes,
                    }
                )

            # Create DataFrame and handle missing values
            features_df = pd.DataFrame(features).fillna(0)

            if len(features_df) == 0:
                return None

            # Create pipeline with imputer
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Transform the data
            X = pipeline.fit_transform(features_df)

            # Only proceed with ML analysis if we have enough data
            if X.shape[0] > 1:
                # PCA
                pca = PCA(n_components=min(2, X.shape[1]))
                X_pca = pca.fit_transform(X)

                # Clustering
                kmeans = KMeans(n_clusters=min(3, X.shape[0]))
                clusters = kmeans.fit_predict(X)

                # Anomaly detection
                iso = IsolationForest()
                anomalies = iso.fit_predict(X)

                return {
                    "pca_components": X_pca.tolist(),
                    "clusters": clusters.tolist(),
                    "anomalies": anomalies.tolist(),
                    "feature_names": list(features_df.columns),
                    "features_importance": dict(
                        zip(features_df.columns, np.abs(pca.components_[0]))
                    ),
                }
            return None

        except Exception as e:
            print(f"Error in pattern analysis: {str(e)}")
            traceback.print_exc()
            return None

    def print_data_status(self):
        print("\nData Manager Status:")
        print(f"Quantum results: {len(self.quantum_results)}")
        if self.quantum_results:
            print(f"Last quantum data: {self.quantum_results[-1]['data'].keys()}")
        print(f"Melody results: {len(self.melody_results)}")
        if self.melody_results:
            print(f"Last melody data: {self.melody_results[-1]['data'].keys()}")
        print(f"Fluid results: {len(self.fluid_results)}")
        if self.fluid_results:
            print(f"Last fluid data: {self.fluid_results[-1]['data'].keys()}")
        print(f"QEC results: {len(self.qec_results)}")
        print(f"Particle results: {len(self.particle_results)}")
