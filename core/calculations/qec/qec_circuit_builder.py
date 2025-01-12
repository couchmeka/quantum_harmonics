from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class QECCircuitBuilder:
    def __init__(self):
        self.qr = None
        self.cr = None

    def create_3qubit_bit_flip(self):
        """Create 3-qubit bit flip code circuit"""
        self.qr = QuantumRegister(3, "q")
        self.cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encoding
        circuit.cx(self.qr[0], self.qr[1])
        circuit.cx(self.qr[0], self.qr[2])

        # Error detection
        circuit.cx(self.qr[0], self.qr[1])
        circuit.cx(self.qr[0], self.qr[2])
        circuit.ccx(self.qr[1], self.qr[2], self.qr[0])

        circuit.measure(self.qr, self.cr)
        return circuit

    def create_3qubit_phase_flip(self):
        """Create 3-qubit phase flip code circuit"""
        self.qr = QuantumRegister(3, "q")
        self.cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(self.qr, self.cr)

        # Apply Hadamard gates for phase flip protection
        circuit.h([0, 1, 2])
        circuit.cx(self.qr[0], self.qr[1])
        circuit.cx(self.qr[0], self.qr[2])
        circuit.h([0, 1, 2])

        circuit.measure(self.qr, self.cr)
        return circuit

    def create_5qubit_code(self):
        """Create 5-qubit perfect code circuit"""
        self.qr = QuantumRegister(5, "q")
        self.cr = ClassicalRegister(5, "c")
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encoding
        circuit.h(self.qr[0])
        circuit.h(self.qr[1])
        circuit.cx(self.qr[1], self.qr[2])
        circuit.cx(self.qr[2], self.qr[3])
        circuit.cx(self.qr[3], self.qr[4])
        circuit.h(self.qr[4])

        circuit.measure(self.qr, self.cr)
        return circuit

    def create_steane_code(self):
        """Create 7-qubit Steane code circuit"""
        self.qr = QuantumRegister(7, "q")
        self.cr = ClassicalRegister(7, "c")
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encoding
        circuit.h(range(7))
        for i in range(6):
            circuit.cx(self.qr[i], self.qr[i + 1])

        circuit.measure(self.qr, self.cr)
        return circuit

    def create_shor_code(self):
        """Create 9-qubit Shor code circuit"""
        self.qr = QuantumRegister(9, "q")
        self.cr = ClassicalRegister(9, "c")
        circuit = QuantumCircuit(self.qr, self.cr)

        # Encoding
        circuit.h([0, 3, 6])
        for i in [0, 3, 6]:
            circuit.cx(self.qr[i], self.qr[i + 1])
            circuit.cx(self.qr[i], self.qr[i + 2])

        circuit.measure(self.qr, self.cr)
        return circuit
