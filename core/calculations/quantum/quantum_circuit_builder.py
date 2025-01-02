# quantum_circuit_builder.py
# This module handles the creation and configuration of quantum circuits for harmonic analysis.
# It provides methods to build circuits with specific frequency and amplitude parameters,
# supporting the quantum harmonic analysis process.
#
# Key Components:
# - Quantum circuit generation for frequency analysis
# - Qubit register setup and measurement configuration
# - Circuit parameter normalization and angle calculations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
import numpy as np
from qiskit_aer import AerSimulator
from qiskit import qasm3


def create_circuit(frequencies, amplitudes):
    """
    Create a quantum circuit for harmonic analysis.

    Parameters:
        frequencies (list): List of frequencies to analyze
        amplitudes (list): Corresponding amplitudes for each frequency

    Returns:
        tuple: (QuantumCircuit, simulation results)
    """
    num_qubits = len(frequencies)
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    circuit = QuantumCircuit(qr, cr)

    # Normalize frequencies and apply quantum gates
    max_freq = max(frequencies)
    for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
        # Calculate rotation angle based on normalized frequency
        angle = np.pi * freq / max_freq * amp
        # Apply rotation gate
        circuit.ry(angle, qr[i])
        # Add entanglement between adjacent qubits
        if i < num_qubits - 1:
            circuit.cx(qr[i], qr[i + 1])

    # Add measurement operations
    circuit.measure(qr, cr)

    # Run simulation
    backend = AerSimulator()
    transpiled_circuit = transpile(circuit, backend, optimization_level=3)
    job = backend.run(transpiled_circuit, shots=1024)
    result = job.result()

    # Return both circuit and simulation results
    return circuit, result
