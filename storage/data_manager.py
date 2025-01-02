# data_manager.py
from datetime import datetime
import copy


class QuantumDataManager:
    _instance = None

    def __init__(self):
        self.fluid_results = []
        self.quantum_results = []
        self.qec_results = []
        self.melody_results = []
        self.circuit_results = []
        self.last_update_time = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QuantumDataManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.fluid_results = None
        self.quantum_results = None
        self.qec_results = None
        self.melody_results = None
        self.last_update_time = None

    def update_fluid_results(self, results):
        self.fluid_results.append(
            {"timestamp": datetime.now(), "data": copy.deepcopy(results)}
        )
        self._update_timestamp()
        print(f"Added fluid results. Total runs: {len(self.fluid_results)}")

    def update_quantum_results(self, results):
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
        self.melody_results.append(
            {"timestamp": datetime.now(), "data": copy.deepcopy(results)}
        )
        self._update_timestamp()
        print(f"Added melody results. Total runs: {len(self.melody_results)}")

    def get_all_results(self):
        return {
            "fluid": self.fluid_results,
            "quantum": self.quantum_results,
            "qec": self.qec_results,
            "melody": self.melody_results,
            "circuit": self.circuit_results,
        }

    def get_latest_results(self):
        return {
            "fluid": self.fluid_results[-1] if self.fluid_results else None,
            "quantum": self.quantum_results[-1] if self.quantum_results else None,
            "qec": self.qec_results[-1] if self.qec_results else None,
            "melody": self.melody_results[-1] if self.melody_results else None,
            "circuit": self.circuit_results[-1] if self.circuit_results else None,
        }

    def clear_all_results(self):
        self.fluid_results = []
        self.quantum_results = []
        self.qec_results = []
        self.melody_results = []
        self.circuit_results = []
        self.last_update_time = None
        print("All results cleared")

    def has_data(self):
        has_data = any(
            [
                self.fluid_results,
                self.quantum_results,
                self.qec_results,
                self.melody_results,
            ]
        )
        print(f"\nChecking for data: {has_data}")
        return has_data

    def _update_timestamp(self):
        self.last_update_time = datetime.now()
        print(f"Updated timestamp: {self.last_update_time}")

    def update_circuit_results(self, results):
        """Update circuit analysis results"""
        print("\nUpdating circuit results in data manager...")
        print(f"Before update: {self.circuit_results is not None}")
        self.circuit_results = copy.deepcopy(results)
        print(f"After update: {self.circuit_results is not None}")
        self._update_timestamp()
