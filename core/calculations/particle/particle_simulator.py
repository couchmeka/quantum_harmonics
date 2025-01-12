# core/calculations/particle/particle_simulator.py
# particle_simulator.py, houses the ParticleSimulator class, which is
# designed to simulate and analyze the dynamics of particle systems
# using both CPU and GPU computation resources. The class supports Apple's Metal Performance Shaders (MPS)
# via PyTorch for GPU-based computations if available. It includes features for
# initializing particle states, computing forces using Newtonian mechanics,
# and updating positions and velocities based on these forces. The class can
# handle both classical particle simulations and applications of quantum mechanical
# properties to explore complex behaviors such as quantum tunneling and decoherence
# effects in a simulated environment. This makes it suitable for research in computational
# physics, material science, and related fields where particle behavior under various forces
# and conditions needs to be studied.


import numpy as np

try:
    import torch
    import torch.mps  # Apple Metal support

    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    torch = None
    MPS_AVAILABLE = False
    print("PyTorch MPS not available, falling back to CPU computation")


class ParticleSimulator:
    def __init__(self, num_particles=5120):
        self.num_particles = num_particles
        self.use_mps = MPS_AVAILABLE

        # Initialize device
        self.device = torch.device("mps") if self.use_mps else "cpu"

        # Initialize particle data
        self.positions = None
        self.velocities = None
        self.accelerations = None
        self.quantum_data = None
        self.reset()

    def reset(self):
        """Initialize or reset particle states"""
        if self.use_mps:
            # Spread particles out more (changed from 0.5 to 2.0)
            self.positions = (
                torch.randn(self.num_particles, 3, device=self.device) * 4.0
            )
            # Increase initial velocities (changed from 0.01 to 0.1)
            self.velocities = (
                torch.randn(self.num_particles, 3, device=self.device) * 0.2
            )
            self.accelerations = torch.zeros(self.num_particles, 3, device=self.device)
        else:
            self.positions = np.random.randn(self.num_particles, 3) * 2.0
            self.velocities = np.random.randn(self.num_particles, 3) * 0.1
            self.accelerations = np.zeros((self.num_particles, 3))

    def compute_forces(self):
        if self.use_mps:
            positions_expanded = self.positions.unsqueeze(1)
            diff = positions_expanded - self.positions.unsqueeze(0)

            distances = torch.norm(diff, dim=2)
            min_distance = 1e-3
            distances = (
                distances
                + torch.eye(self.num_particles, device=self.device) * min_distance
            )
            distances = torch.clamp(distances, min=min_distance)

            # Increase force strength
            force_magnitudes = 1e-1 / (distances**3)  # Changed from 1e-2
            force_magnitudes = force_magnitudes * (
                1 - torch.eye(self.num_particles, device=self.device)
            )

            forces = force_magnitudes.unsqueeze(2) * diff
            accelerations = forces.sum(dim=0)

            return accelerations

        # CPU computation
        positions_expanded = self.positions[:, np.newaxis]
        diff = positions_expanded - self.positions

        distances = np.linalg.norm(diff, axis=2)
        min_distance = np.full_like(
            distances, 1e-3
        )  # Use numpy array instead of scalar
        distances = np.maximum(distances, min_distance)
        distances += np.eye(self.num_particles) * min_distance[0]

        force_magnitudes = 1e-2 / (distances**3)
        force_magnitudes = np.where(
            np.isfinite(force_magnitudes), force_magnitudes, 0.0
        )
        force_magnitudes *= 1 - np.eye(self.num_particles)

        forces = force_magnitudes[:, :, np.newaxis] * diff
        accelerations = np.where(
            np.isfinite(forces.sum(axis=0)), forces.sum(axis=0), 0.0
        )

        return accelerations

    def update(self, dt=0.01, fluid_forces=None):
        """Update particle positions and velocities"""
        # Compute accelerations
        self.accelerations = self.compute_forces()

        if fluid_forces is not None:
            self.accelerations += fluid_forces

        if self.use_mps:
            # GPU version
            # Update velocities (half-step)
            self.velocities += 0.5 * self.accelerations * dt * 2.0

            # Update positions
            self.positions += self.velocities * dt * 2.0

            # Recompute accelerations
            self.accelerations = self.compute_forces()

            # Add stronger random perturbations
            if torch.rand(1).item() < 0.1:
                self.velocities += torch.randn_like(self.velocities) * 0.3

            # Reduce damping
            self.velocities *= 0.995

            # Keep particles within bounds
            self.positions = torch.clamp(self.positions, -8, 8)

            # Scale positions for visualization
            scaled_positions = self.positions * 2

            # Convert to NumPy for visualization
            return (
                scaled_positions.cpu().numpy(),
                self.velocities.cpu().numpy(),
                self.accelerations.cpu().numpy(),
            )
        else:
            # CPU version
            # Update velocities (half-step)
            self.velocities += 0.5 * self.accelerations * dt * 2.0

            # Update positions
            self.positions += self.velocities * dt * 2.0

            # Recompute accelerations
            self.accelerations = self.compute_forces()

            # Update velocities (half-step)
            self.velocities += 0.5 * self.accelerations * dt * 2.0

            # Add stronger random perturbations
            if np.random.random() < 0.1:
                self.velocities += np.random.randn(*self.velocities.shape) * 0.3

            # Reduce damping
            self.velocities *= 0.995

            # Keep particles within bounds
            self.positions = np.clip(self.positions, -8, 8)

            # Scale positions for visualization
            scaled_positions = self.positions * 2

            return scaled_positions, self.velocities, self.accelerations

    def get_quantum_states(self):
        """Get quantum state representation"""
        if self.use_mps:
            psi = self.positions[:, 0] + 1j * self.velocities[:, 0]
            psi = psi.cpu().numpy()  # Convert to NumPy
        else:
            psi = self.positions[:, 0] + 1j * self.velocities[:, 0]

        psi /= np.linalg.norm(psi)
        return psi

    def apply_quantum_data(self, quantum_data):
        """Apply quantum melody analysis data to particle simulation"""
        self.quantum_data = quantum_data
        if not self.quantum_data:
            return

        # Add missing data with defaults
        quantum_data.update(
            {
                "quantum_metrics": quantum_data.get(
                    "quantum_metrics",
                    {
                        "purity": quantum_data.get("purity", 1.0),
                        "fidelity": quantum_data.get("fidelity", 1.0),
                    },
                ),
                "phases": (
                    np.angle(quantum_data.get("statevector", []))
                    if "statevector" in quantum_data
                    else np.zeros(len(quantum_data.get("frequencies", [1.0])))
                ),
                "pythagorean_analysis": quantum_data.get(
                    "pythagorean_analysis", [{"harmonic_influence": 1.0}]
                ),
            }
        )

        # Extract properties
        frequencies = quantum_data["quantum_frequencies"]
        phases = quantum_data["phases"]
        pythagorean = quantum_data["pythagorean_analysis"]

        # Reset particles
        self.num_particles = 1024
        self.reset()

        try:
            if self.use_mps:
                # MPS version with float32
                phase_field = torch.tensor(
                    phases, device=self.device, dtype=torch.float32
                ).repeat((self.num_particles + len(phases) - 1) // len(phases))[
                    : self.num_particles
                ]

                freq_field = torch.tensor(
                    frequencies, device=self.device, dtype=torch.float32
                ).repeat(
                    (self.num_particles + len(frequencies) - 1) // len(frequencies)
                )[
                    : self.num_particles
                ]

                harmonic_values = [float(p["harmonic_influence"]) for p in pythagorean]
                harmonic_field = torch.tensor(
                    harmonic_values, device=self.device, dtype=torch.float32
                ).repeat(
                    (self.num_particles + len(harmonic_values) - 1)
                    // len(harmonic_values)
                )[
                    : self.num_particles
                ]

                self.positions *= 1 + 0.2 * phase_field.unsqueeze(1)
                self.velocities *= 1 + 0.1 * freq_field.unsqueeze(1)
                self.velocities += 0.1 * harmonic_field.unsqueeze(1)

            else:
                # CPU version
                phase_field = np.tile(
                    phases, (self.num_particles + len(phases) - 1) // len(phases)
                )[: self.num_particles]
                freq_field = np.tile(
                    frequencies,
                    (self.num_particles + len(frequencies) - 1) // len(frequencies),
                )[: self.num_particles]
                harmonic_values = [p["harmonic_influence"] for p in pythagorean]
                harmonic_field = np.tile(
                    harmonic_values,
                    (self.num_particles + len(harmonic_values) - 1)
                    // len(harmonic_values),
                )[: self.num_particles]

                self.positions *= 1 + 0.2 * phase_field[:, np.newaxis]
                self.velocities *= 1 + 0.1 * freq_field[:, np.newaxis]
                self.velocities += 0.1 * harmonic_field[:, np.newaxis]

        except Exception as e:
            print(f"Error in quantum influence application: {str(e)}")
            self.reset()
