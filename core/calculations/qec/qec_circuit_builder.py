from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class QECCircuitBuilder:
    def __init__(self):
        self.qr = None
        self.cr = None

    def create_3qubit_bit_flip(self):
        self.qr = QuantumRegister(3, "q")
        self.cr = ClassicalRegister(2, "c")  # Measure only ancillary qubits
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encoding
        circuit.cx(self.qr[0], self.qr[1])
        circuit.cx(self.qr[0], self.qr[2])

        # Error simulation (comment this for real implementation)
        # circuit.x(self.qr[1])  # Simulating bit-flip error on qubit 1

        # Error detection and correction
        circuit.cx(self.qr[0], self.qr[1])
        circuit.cx(self.qr[0], self.qr[2])
        circuit.ccx(self.qr[1], self.qr[2], self.qr[0])  # Correction

        # Measure ancillary qubits
        circuit.measure([1, 2], self.cr)

        return circuit

    def create_3qubit_phase_flip(self):
        self.qr = QuantumRegister(3, "q")
        self.cr = ClassicalRegister(2, "c")  # Measure only ancillary qubits
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encoding
        circuit.h([0, 1, 2])
        circuit.cx(self.qr[0], self.qr[1])
        circuit.cx(self.qr[0], self.qr[2])

        # Error simulation (comment this for real implementation)
        # circuit.z(self.qr[1])  # Simulating phase-flip error on qubit 1

        # Decoding
        circuit.h([0, 1, 2])

        # Measure ancillary qubits
        circuit.measure([1, 2], self.cr)

        return circuit

    def create_5qubit_code(self):
        self.qr = QuantumRegister(5, "q")
        self.cr = ClassicalRegister(4, "c")  # Measure stabilizer qubits
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encoding (simplified for clarity; adjust for actual stabilizers)
        circuit.h(self.qr[0])
        circuit.cx(self.qr[0], self.qr[1])
        circuit.cx(self.qr[1], self.qr[2])
        circuit.cx(self.qr[2], self.qr[3])
        circuit.cx(self.qr[3], self.qr[4])

        # Error simulation (comment this for real implementation)
        # circuit.x(self.qr[2])  # Simulating bit-flip error on qubit 2

        # Measure stabilizers
        circuit.measure([1, 2, 3, 4], self.cr)

        return circuit

    def create_steane_code(self):
        self.qr = QuantumRegister(7, "q")
        self.cr = ClassicalRegister(3, "c")  # Measure stabilizer syndromes
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encoding
        circuit.h(self.qr[0])
        circuit.cx(self.qr[0], self.qr[1])
        circuit.cx(self.qr[0], self.qr[2])
        circuit.cx(self.qr[1], self.qr[3])
        circuit.cx(self.qr[2], self.qr[4])
        circuit.cx(self.qr[3], self.qr[5])
        circuit.cx(self.qr[4], self.qr[6])

        # Measure stabilizers (syndromes)
        circuit.measure([3, 5, 6], self.cr)

        return circuit

    def create_shor_code(self):
        self.qr = QuantumRegister(9, "q")
        self.cr = ClassicalRegister(6, "c")  # Measure stabilizers
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encoding
        for i in [0, 3, 6]:
            circuit.h(self.qr[i])
            circuit.cx(self.qr[i], self.qr[i + 1])
            circuit.cx(self.qr[i], self.qr[i + 2])

        # Measure stabilizers
        circuit.measure([1, 4, 7, 2, 5, 8], self.cr)

        return circuit
