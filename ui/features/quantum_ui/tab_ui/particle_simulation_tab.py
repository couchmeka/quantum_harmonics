import traceback

import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QComboBox,
    QTextEdit,
    QTabWidget,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QMutex, QMutexLocker
from matplotlib import pyplot as plt, animation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from core.calculations.particle.particle_simulator import ParticleSimulator
from data.backend_data_management.data_manager import QuantumDataManager
from core.calculations.particle.bloch_sphere_visualizer import BlochSphereVisualizer
from qiskit.visualization import plot_bloch_multivector
from qiskit.visualization.exceptions import VisualizationError
from qiskit.quantum_info import Statevector
from ui.styles_ui.styles import *


class ParticleSimulationTab(QWidget):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, data_manager=None):
        super().__init__()
        self.time_label = None
        self.step_back_btn = None
        self.viz_style = None
        self.step_forward_btn = None
        self.time = None
        self.velocities = None
        self.accelerations = None
        self.bloch_visualizer = None
        self._animation_frame = None
        self._animation_frame = 0
        if not data_manager:
            raise ValueError("data_manager is required")
        self.data_manager = data_manager or QuantumDataManager()
        self.data_manager = data_manager
        self._canvas_mutex = QMutex()
        # Add error tracking
        self.error_count = 0
        self.max_errors = 3  # Maximum number of errors before stopping simulation
        self.speed_label = None
        self._scatter = None
        self.colorbar = None
        self.toggle_button = None
        self.viz_mode = None
        self.speed_slider = None
        self.data_source = None
        self.animation_speed = 20
        self.simulator = ParticleSimulator(num_particles=1024)
        self.current_mode = "Energy"
        self.quantum_data_text = None
        self.import_fluid_btn = None
        self.fluid_data = None
        self.using_fluid_data = False
        self.import_quantum_btn = None
        self.quantum_data = None

        try:
            self.figure = Figure(figsize=(8, 8))
            self.figure.patch.set_facecolor(COLORS["background"])
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111, projection="3d")
            self.ax.set_facecolor(COLORS["background"])
        except Exception as e:
            print(f"Error initializing matplotlib: {str(e)}")
            # If initialization fails, re-attempt to create the objects
            self.figure = Figure(figsize=(8, 8))
            self.figure.patch.set_facecolor(COLORS["background"])
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111, projection="3d")
            self.ax.set_facecolor(COLORS["background"])
            print("Re-initialized Matplotlib objects.")

        # Animation control
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.setInterval(self.animation_speed)

        # Connect to fluid data manager updates
        self.check_fluid_data_timer = QTimer()
        self.check_fluid_data_timer.timeout.connect(self.check_fluid_data_updates)
        self.check_fluid_data_timer.start(1000)

        self.check_quantum_data_timer = QTimer()
        self.check_quantum_data_timer.timeout.connect(self.check_quantum_data_updates)
        self.check_quantum_data_timer.start(1000)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 10)
        layout.setSpacing(5)

        # Initialize components
        self.import_quantum_btn = create_button("Import Quantum Data")
        self.import_quantum_btn.setToolTip("Import data from Quantum Analysis")
        self.import_quantum_btn.clicked.connect(self.import_quantum_data)

        self.import_fluid_btn = create_button("Import Fluid Data")
        self.import_fluid_btn.setToolTip("Import data from Fluid Dynamics")
        self.import_fluid_btn.clicked.connect(self.import_fluid_data)

        self.data_source = QComboBox()
        self.data_source.addItems(["Particle Simulation", "Quantum Data", "Fluid Data"])
        self.data_source.setToolTip("Select data source for particle behavior")
        self.data_source.currentTextChanged.connect(self.change_data_source)

        self.viz_mode = QComboBox()
        self.viz_mode.addItems(
            ["Velocity", "Position", "Energy", "Quantum State", "Spectral Lines"]
        )
        self.viz_mode.setToolTip("Select visualization type")
        self.viz_mode.currentTextChanged.connect(self.update_visualization)

        # In setup_ui method, add this after self.viz_mode combobox setup:
        self.viz_style = QComboBox()
        self.viz_style.addItems(["Particle Simulation", "Bloch Sphere"])
        self.viz_style.setToolTip("Select visualization style")
        self.viz_style.currentTextChanged.connect(self.change_visualization_style)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(20, 500)
        self.speed_slider.setValue(100)
        self.speed_slider.setToolTip("Adjust simulation speed")
        self.speed_slider.valueChanged.connect(self.update_speed)

        self.speed_label = QLabel("100")
        self.speed_label.setStyleSheet(base_style)

        self.toggle_button = create_button("Start")
        self.toggle_button.setToolTip("Start/Stop simulation")
        self.toggle_button.clicked.connect(self.toggle_simulation)

        self.quantum_data_text = QTextEdit()
        self.quantum_data_text.setReadOnly(True)
        self.quantum_data_text.setStyleSheet(textedit_style)

        # Top Controls
        controls_group = create_group_box("Particle Controls")
        controls_layout = QVBoxLayout()

        # First row of controls
        top_controls = QHBoxLayout()
        top_controls.addWidget(self.import_quantum_btn)
        top_controls.addWidget(self.import_fluid_btn)

        source_label = QLabel("Data Source:")
        source_label.setStyleSheet(base_style)
        top_controls.addWidget(source_label)
        top_controls.addWidget(self.data_source)

        reset_button = create_button("Reset")
        reset_button.setToolTip("Reset simulation to initial state")
        reset_button.clicked.connect(self.reset_simulation)
        top_controls.addWidget(reset_button)
        top_controls.addWidget(self.toggle_button)
        controls_layout.addLayout(top_controls)

        # Second row of controls
        viz_controls = QHBoxLayout()
        viz_label = QLabel("Visualization Mode:")
        viz_label.setStyleSheet(base_style)
        viz_controls.addWidget(viz_label)
        viz_controls.addWidget(self.viz_mode)
        # Add the new combobox to viz_controls layout
        viz_controls.addWidget(QLabel("Visualization Style:"))
        viz_controls.addWidget(self.viz_style)

        speed_layout = QHBoxLayout()
        speed_label = QLabel("Animation Speed:")
        speed_label.setStyleSheet(base_style)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)
        viz_controls.addLayout(speed_layout)
        controls_layout.addLayout(viz_controls)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Main content area
        content_container = QWidget()
        content_layout = QHBoxLayout(content_container)

        # Left: Visualization
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        canvas_group = create_group_box("Visualization")
        canvas_layout = QVBoxLayout()
        canvas_layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)
        self.canvas.setMinimumHeight(400)
        canvas_layout.setContentsMargins(0, 0, 0, 20)
        canvas_group.setLayout(canvas_layout)
        viz_layout.addWidget(canvas_group)
        content_layout.addWidget(viz_container, stretch=5)

        # Right: Analysis Results
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        results_label = QLabel("Analysis Results")
        results_label.setStyleSheet("color: white; font-weight: bold;")
        self.quantum_data_text.setStyleSheet(
            textedit_style
            + """
           QTextEdit {
               min-width: 300px;
               max-width: 400px;
           }
       """
        )
        results_layout.addWidget(results_label)
        results_layout.addWidget(self.quantum_data_text)
        results_container.setLayout(results_layout)
        content_layout.addWidget(results_container, stretch=1)

        layout.addWidget(content_container)
        self.setLayout(layout)

        # Time step controls
        time_controls = QHBoxLayout()

        # Step backward button
        self.step_back_btn = create_button("◀")
        self.step_back_btn.setToolTip("Step backward")
        self.step_back_btn.clicked.connect(self.step_backward)
        time_controls.addWidget(self.step_back_btn)

        # Time slider
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)  # Will be updated when data loads
        self.time_slider.setValue(0)
        self.time_slider.setToolTip("Time step")
        # In setup_ui, change this line:
        self.time_slider.valueChanged.connect(
            self.update_timestep
        )  # Changed from update_time_step
        time_controls.addWidget(self.time_slider)

        # Step forward button
        self.step_forward_btn = create_button("▶")
        self.step_forward_btn.setToolTip("Step forward")
        self.step_forward_btn.clicked.connect(self.step_forward)
        time_controls.addWidget(self.step_forward_btn)

        # Time step display
        self.time_label = QLabel("Step: 0")
        self.time_label.setStyleSheet(base_style)
        time_controls.addWidget(self.time_label)

        # Add to controls layout
        controls_layout.addLayout(time_controls)

    def update_speed(self, value):
        """Update animation speed and label"""
        self.animation_speed = 3000 // value
        self.speed_label.setText(str(value))
        if self.timer.isActive():
            self.timer.setInterval(self.animation_speed)

    # In ParticleSimulationTab:
    def import_quantum_data(self, quantum_data):
        """Import and process quantum data"""
        try:
            if not quantum_data:
                print("No quantum data provided")
                return

            print(f"\nDEBUG - Processing Quantum Data:")
            print(f"Received data with keys: {quantum_data.keys()}")

            # Extract and validate quantum frequencies
            quantum_frequencies = quantum_data.get("quantum_frequencies", [])
            if not quantum_frequencies:
                quantum_frequencies = quantum_data.get(
                    "frequencies", []
                )  # Try alternate key

            print(f"Found quantum frequencies: {quantum_frequencies}")

            # Extract other essential data
            statevector = quantum_data.get("statevector", None)
            purity = quantum_data.get("purity", 0)
            fidelity = quantum_data.get("fidelity", 0)
            phases = quantum_data.get("phases", [])
            amplitudes = quantum_data.get("amplitudes", [])

            print(f"Found {len(quantum_frequencies)} frequencies")
            print(f"Statevector present: {'Yes' if statevector is not None else 'No'}")

            # Store in simulator with frequencies
            self.simulator.quantum_data = {
                "quantum_frequencies": quantum_frequencies,
                "statevector": statevector,
                "purity": purity,
                "fidelity": fidelity,
                "phases": phases,
                "amplitudes": amplitudes,
            }

            metrics = {
                "quantum_frequencies": quantum_frequencies,
                "quantum_metrics": {"purity": purity, "fidelity": fidelity},
                "phases": phases,
                "amplitudes": amplitudes,
            }

            if "harmony_data" in quantum_data:
                metrics["harmony_data"] = quantum_data["harmony_data"]

            self._update_quantum_display(metrics)
            print("Successfully processed quantum data")

        except Exception as e:
            print(f"Error importing quantum data: {str(e)}")
            traceback.print_exc()

    def apply_quantum_state(self, statevector):
        """Apply quantum state to particle simulation with enhanced compatibility"""
        try:
            if statevector is None:
                return

            # Handle both real and complex statevectors
            if np.iscomplexobj(statevector):
                statevector = np.abs(statevector)  # Use magnitude for visualization

            # Normalize the statevector
            if np.sum(statevector) > 0:
                statevector = statevector / np.max(statevector)

            # Map to particle positions
            n_particles = len(self.positions)
            state_len = len(statevector)

            # Repeat the statevector if needed
            indices = np.arange(n_particles) % state_len
            particle_states = statevector[indices]

            # Scale factor for visualization
            scale = 10.0

            # Apply to particle positions
            for i in range(min(3, self.positions.shape[1])):  # Up to 3 dimensions
                self.positions[:, i] = (
                    particle_states * scale * np.cos(2 * np.pi * i / 3)
                )
                self.velocities[:, i] = (
                    particle_states * scale / 2 * np.sin(2 * np.pi * i / 3)
                )

        except Exception as e:
            print(f"Error applying quantum state: {str(e)}")
            traceback.print_exc()

    def apply_quantum_oscillations(self, frequencies, phases):
        """Apply quantum oscillations with enhanced compatibility"""
        try:
            if frequencies is None or phases is None:
                return

            # Normalize frequencies to usable range
            norm_freqs = (
                frequencies / np.max(frequencies)
                if np.max(frequencies) > 0
                else frequencies
            )

            # Map to particles
            particle_indices = np.arange(self.num_particles) % len(frequencies)
            self.oscillation_freqs = norm_freqs[particle_indices]
            self.oscillation_phases = phases[particle_indices]

            # Apply initial modifications
            phase_factor = np.cos(self.oscillation_phases)[:, np.newaxis]
            self.positions *= 1 + 0.2 * phase_factor
            self.velocities *= 1 + 0.1 * phase_factor

            # Store for animation updates
            self.base_frequencies = frequencies[particle_indices]

        except Exception as e:
            print(f"Error applying quantum oscillations: {str(e)}")
            traceback.print_exc()

    def change_data_source(self, source):
        """Handle data source changes"""
        if source == "Quantum Data" and not self.simulator.quantum_data:
            self.quantum_data_text.setText(
                "No quantum data loaded. Please import data first."
            )
            self.data_source.setCurrentText("Particle Simulation")
        self.update_visualization(self.viz_mode.currentText())

    def update_visualization(self, mode):
        """Update visualization based on selected mode"""
        self.current_mode = mode
        # Update colorbar if it exists
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        self.update_simulation()

        # After updating simulation, update colorbar label
        if self.colorbar is not None:
            self.colorbar.set_label(self.current_mode, color="white")
            self.colorbar.ax.yaxis.set_tick_params(color="white")
            self.colorbar.ax.yaxis.label.set_color("white")

    def toggle_simulation(self):
        """Toggle simulation start/stop"""
        if self.timer.isActive():
            self.timer.stop()
            self.toggle_button.setText("Start")
        else:
            self.timer.start(self.animation_speed)
            self.toggle_button.setText("Stop")

    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.simulator.reset()
        self.update_simulation()
        self.quantum_data_text.clear()
        self.data_source.setCurrentText("Particle Simulation")

    def reset(self):
        """Reset particles to initial state"""
        self.positions = np.random.randn(self.num_particles, 3) * 5
        self.velocities = np.random.randn(self.num_particles, 3) * 0.1
        self.accelerations = np.zeros((self.num_particles, 3))
        self.time = 0.0

    def update(self, dt=None):
        """Update particle positions and velocities"""
        if dt is not None:
            self.dt = dt

        # Apply quantum oscillations if available
        if hasattr(self, "base_frequencies") and hasattr(self, "oscillation_phases"):
            time_factor = np.cos(
                2 * np.pi * self.base_frequencies * self.time + self.oscillation_phases
            )
            self.velocities += 0.1 * time_factor[:, np.newaxis] * self.positions

        # Update positions and velocities
        self.positions += self.velocities * self.dt
        self.velocities += self.accelerations * self.dt

        # Add some damping
        self.velocities *= 0.99

        # Keep particles in bounds
        bound = 15.0
        self.positions = np.clip(self.positions, -bound, bound)

        self.time += self.dt
        return self.positions, self.velocities, self.accelerations

    # This is where storage is stored
    def update_simulation(self):
        try:
            with QMutexLocker(self._canvas_mutex):
                if not self._check_widget_existence():
                    return

                positions, velocities, accelerations = self.simulator.update(dt=0.01)

                if not self._check_widget_visibility():
                    return

                self._update_visualization(positions, velocities)
                self.canvas.draw_idle()  # Use draw_idle

        except RuntimeError as e:
            if "deleted" in str(e):
                print("Widget has been deleted, stopping simulation.")
                self.timer.stop()
            else:
                print(f"RuntimeError encountered: {str(e)}")

    # Helper methods
    def _check_widget_existence(self):
        """Check if critical widgets still exist"""
        if not hasattr(self, "canvas") or not hasattr(self, "toggle_button"):
            print("Widgets have been cleaned up, stopping simulation.")
            self._stop_timer()
            return False
        return True

    def _check_widget_visibility(self):
        """Check if widget is visible and not hidden"""
        if not self.isVisible() or self.isHidden():
            return False
        return True

    def _update_visualization(self, positions, velocities):
        """Update the scatter plot and colorbar"""
        try:
            # Remove colorbar if it exists
            if self.colorbar is not None:
                self.colorbar.remove()
                self.colorbar = None

            # Create scatter plot if it doesn't exist
            if not self._scatter:
                self._initialize_scatter(positions, velocities)
            else:
                self._update_scatter(positions, velocities)

            # Add colorbar if needed
            if self.colorbar is None and hasattr(self, "figure"):
                self._add_colorbar()

            # Redraw canvas if it exists
            if hasattr(self, "canvas") and self.canvas and self.canvas.isVisible():
                self.canvas.draw()

        except RuntimeError as e:
            if "deleted" in str(e):
                print("Canvas or scatter plot deleted, stopping simulation.")
                self._stop_timer()
                return
            raise

    def _initialize_scatter(self, positions, velocities):
        """Initialize scatter plot"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.set_facecolor(COLORS["background"])
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.set_zlim(-15, 15)
        self.ax.grid(True)
        self.ax.set_position([0.1, 0.1, 0.8, 0.8])

        # Customize axis labels and ticks
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.zaxis.label.set_color("white")
        self.ax.tick_params(colors="white")

        colors = self._get_colors(positions, velocities)
        self._scatter = self.ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=colors,
            cmap="viridis",
            alpha=0.6,
        )

    def _update_scatter(self, positions, velocities):
        """Update scatter plot with new data"""
        self._scatter._offsets3d = (
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
        )
        colors = self._get_colors(positions, velocities)
        self._scatter.set_array(colors)

    def _add_colorbar(self):
        """Add a colorbar to the plot"""
        self.colorbar = self.figure.colorbar(self._scatter)
        self.colorbar.set_label(self.current_mode, color="white")
        self.colorbar.ax.yaxis.set_tick_params(colors="white")
        self.colorbar.ax.yaxis.label.set_color("white")
        plt.setp(plt.getp(self.colorbar.ax.axes, "yticklabels"), color="white")

    def _stop_timer(self):
        """Stop the simulation timer"""
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()

    def _update_quantum_display(self, quantum_data):
        """Helper method to update the quantum data display"""
        try:
            display_text = []

            # Basic quantum metrics
            if "quantum_metrics" in quantum_data:
                metrics = quantum_data["quantum_metrics"]
                display_text.append("=== Quantum State Metrics ===")
                display_text.append(f"Purity: {metrics['purity']:.3f}")
                display_text.append(f"Fidelity: {metrics['fidelity']:.3f}")

            # Frequencies and phases
            freqs = quantum_data.get("quantum_frequencies", [])
            phases = quantum_data.get("phases", [])
            if freqs:
                display_text.append("\n=== Frequency Analysis ===")
                display_text.append(f"Number of frequencies: {len(freqs)}")
                display_text.append(
                    f"Frequency range: {min(freqs):.2f} Hz - {max(freqs):.2f} Hz"
                )

            if phases:
                display_text.append(f"Phase components: {len(phases)}")

            if self.viz_style and self.viz_style.currentText() == "Bloch Sphere":
                display_text.append("=== Bloch Sphere Visualization Guide ===")
                display_text.append("• Red dot: Current quantum state")
                display_text.append("• Green line: Path of state evolution")
                display_text.append("• Sphere coordinates show quantum superposition:")
                display_text.append("  - North pole (z=1): Pure |0⟩ state")
                display_text.append("  - South pole (z=-1): Pure |1⟩ state")
                display_text.append("  - Equator: Equal superposition")
                display_text.append("  - Front (x=1): |+⟩ state")
                display_text.append("  - Back (x=-1): |-⟩ state")
                display_text.append("  - Right (y=1): |i+⟩ state")
                display_text.append("  - Left (y=-1): |i-⟩ state")
                display_text.append("\nCurrent state coordinates are shown on the plot")
            # Bloch sphere coordinates
            if self.viz_style and self.viz_style.currentText() == "Bloch Sphere":
                display_text.append("\n=== Bloch Sphere State ===")
                display_text.append(
                    "The Bloch sphere represents a quantum state as a point on a unit sphere:"
                )
                display_text.append("• |0⟩ state is at the north pole (z = 1)")
                display_text.append("• |1⟩ state is at the south pole (z = -1)")
                display_text.append("• Superposition states lie between poles")
                display_text.append("• The green trajectory shows state evolution")
                display_text.append("\nCurrent coordinates shown in state text")

            # Musical systems if present
            if "harmony_data" in quantum_data:
                display_text.append("\n=== Musical Analysis ===")
                for system, notes in (
                    quantum_data["harmony_data"].get("musical_systems", {}).items()
                ):
                    display_text.append(f"{system}: {', '.join(notes)}")

            self.quantum_data_text.setText("\n".join(display_text))
            self.data_source.setCurrentText("Quantum Data")

        except Exception as e:
            self.quantum_data_text.setText(f"Error updating quantum display: {str(e)}")

    def _get_colors(self, positions, velocities):
        try:
            if self.current_mode == "Quantum State":
                if (
                    hasattr(self.simulator, "quantum_data")
                    and self.simulator.quantum_data is not None
                ):
                    quantum_data = self.simulator.quantum_data

                    if (
                        "statevector" in quantum_data
                        and quantum_data["statevector"] is not None
                    ):
                        statevector = np.array(quantum_data["statevector"])
                        if statevector.size > 0:
                            if len(statevector) != len(positions):
                                indices = np.arange(len(positions)) % len(statevector)
                                statevector = statevector[indices]
                            values = np.abs(statevector)
                            if values.size > 0 and np.max(values) > 0:
                                return values / np.max(values)
                    elif "quantum_frequencies" in quantum_data:
                        freqs = np.array(quantum_data["quantum_frequencies"])
                        if freqs.size > 0:
                            statevector = freqs / np.sqrt(np.sum(freqs**2))
                            if len(statevector) != len(positions):
                                indices = np.arange(len(positions)) % len(statevector)
                                statevector = statevector[indices]
                            values = np.abs(statevector)
                            if values.size > 0 and np.max(values) > 0:
                                return values / np.max(values)
                return np.zeros(len(positions))

            elif self.current_mode == "Velocity":
                return np.linalg.norm(velocities, axis=1)
            elif self.current_mode == "Position":
                return np.linalg.norm(positions, axis=1)
            elif self.current_mode == "Energy":
                return 0.5 * np.sum(velocities**2, axis=1)
            elif self.current_mode == "Spectral Lines":
                hist = self._calculate_spectral_lines(positions, velocities)
                if hist is not None and isinstance(hist, np.ndarray) and hist.size > 0:
                    return hist
                return np.zeros(len(positions))

            return np.zeros(len(positions))

        except Exception as e:
            print(f"Error getting colors: {str(e)}")
            return np.zeros(len(positions))

    def _calculate_spectral_lines(self, positions, velocities):
        kinetic = 0.5 * np.sum(velocities**2, axis=1)
        potential = np.sum(positions**2, axis=1)
        total_energy = kinetic + potential
        hist, _ = np.histogram(total_energy, bins=len(positions), density=True)
        return hist

    def _update_spectral_display(self):
        """Update the spectral visualization"""
        if self.current_mode == "Spectral Lines":
            self.ax.set_xlabel("Energy Level")
            self.ax.set_ylabel("Intensity")
            self.ax.set_zlabel("Height")
            self.ax.set_xlim(-15, 15)
            self.ax.set_ylim(-15, 15)
            self.ax.set_zlim(-15, 15)

    def closeEvent(self, event):
        try:
            # Stop all animations first
            if hasattr(self, "timer") and self.timer is not None:
                self.timer.stop()
                self.timer.timeout.disconnect()

            with QMutexLocker(self._canvas_mutex):
                # Clean up matplotlib resources
                if hasattr(self, "figure") and self.figure is not None:
                    plt.close(self.figure)
                    self.figure = None

                # Clean up Qt resources
                if hasattr(self, "canvas") and self.canvas is not None:
                    self.canvas.close()

                # Clean up Bloch visualizer
                if (
                    hasattr(self, "bloch_visualizer")
                    and self.bloch_visualizer is not None
                ):
                    self.bloch_visualizer.cleanup()
                    self.bloch_visualizer = None

            # Finally delete the canvas
            if hasattr(self, "canvas") and self.canvas is not None:
                self.canvas.deleteLater()
                self.canvas = None

            super().closeEvent(event)
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def hideEvent(self, event):
        """Handle cleanup when widget is hidden"""
        if hasattr(self, "timer") and self.timer is not None:
            self.timer.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        """Handle setup when widget is shown"""
        super().showEvent(event)
        # Only restart if it was previously running
        if (
            hasattr(self, "toggle_button")
            and self.toggle_button is not None
            and self.toggle_button.text() == "Stop"
        ):
            if hasattr(self, "timer") and self.timer is not None:
                self.timer.start(self.animation_speed)

    def _cleanup_qt_objects(self):
        if hasattr(self, "timer") and self.timer is not None:
            self.timer.stop()
            self.timer.deleteLater()
        if hasattr(self, "bloch_visualizer") and self.bloch_visualizer is not None:
            self.bloch_visualizer.cleanup()  # Call cleanup here
            self.bloch_visualizer = None
        if hasattr(self, "figure") and self.figure is not None:
            plt.close(self.figure)

    def safe_update_simulation(self):
        """Wrapper for update_simulation with better error handling"""
        try:
            if not hasattr(self, "simulator") or self.simulator is None:
                print("Warning: Simulator not initialized")
                self.timer.stop()
                return

            self.update_simulation()
            self.error_count = 0  # Reset error count on successful update
        except Exception as e:
            self.error_count += 1
            print(f"Error in simulation update: {str(e)}")
            if self.error_count >= self.max_errors:
                print("Too many errors, stopping simulation")
                self.timer.stop()
                self.toggle_button.setText("Start")
                self.error_count = 0
            traceback.print_exc()

    def check_fluid_data_updates(self):
        """Check for new fluid data updates"""
        if self.data_manager and self.data_manager.fluid_results:
            latest_fluid = self.data_manager.fluid_results[-1]
            if latest_fluid != self.fluid_data:
                self.fluid_data = latest_fluid
                if self.using_fluid_data:
                    self.apply_fluid_data(self.fluid_data)

    def check_quantum_data_updates(self):
        """Check for new quantum data updates"""
        try:
            if self.data_manager and self.data_manager.quantum_results:
                latest_quantum = self.data_manager.quantum_results[-1]
                current_data_id = id(latest_quantum)

                if current_data_id != getattr(self, "_last_quantum_data_id", None):
                    print("\nNew quantum data detected")

                    if "data" in latest_quantum:
                        self.quantum_data = latest_quantum
                        self.import_quantum_data(latest_quantum["data"])

                        # Update Bloch visualization if active
                        if (
                            hasattr(self, "viz_style")
                            and self.viz_style.currentText() == "Bloch Sphere"
                            and hasattr(self, "bloch_visualizer")
                        ):
                            self.bloch_visualizer.calculate_bloch_coordinates(
                                latest_quantum["data"]
                            )

                    self._last_quantum_data_id = current_data_id

        except Exception as e:
            print(f"Error checking quantum data updates: {str(e)}")
            traceback.print_exc()

    def apply_fluid_data(self, fluid_data):
        try:
            if not fluid_data or "data" not in fluid_data:
                return

            data = fluid_data["data"]
            solution = None  # Initialize solution
            if "solution" in data and isinstance(data["solution"], list):
                solution = np.array(data["solution"])

                if solution.shape[1] >= 3:
                    velocities = solution[:, :3]
                    max_vel = np.max(np.abs(velocities))
                    if max_vel > 0:
                        scaled_velocities = velocities / max_vel * 10
                        self.simulator.set_velocities(scaled_velocities)

            # Update quantum data display
            self._update_quantum_display(
                {
                    "frequencies": data.get("original_frequencies", []),
                    "fibonacci_frequencies": data.get("fibonacci_frequencies", []),
                    "quantum_metrics": {
                        "energy": (
                            np.mean(solution[:, 3])
                            if solution is not None and solution.shape[1] > 3
                            else 0
                        )
                    },
                }
            )

        except Exception as e:
            print(f"Error applying fluid data: {str(e)}")

    def import_fluid_data(self):
        """Import data from fluid dynamics tab with proper error handling"""
        try:
            if not self.data_manager.fluid_results:
                self.quantum_data_text.setText(
                    "No fluid data available in data manager"
                )
                return

            latest_fluid = self.data_manager.fluid_results[-1]
            if latest_fluid and "data" in latest_fluid:
                self.fluid_data = latest_fluid
                self.using_fluid_data = True
                self.data_source.setCurrentText("Fluid Data")
                self.apply_fluid_data(self.fluid_data)

                # Update display
                self.quantum_data_text.setText(
                    "Fluid data imported successfully\n"
                    f"Number of time points: {len(self.fluid_data['data'].get('t', []))}\n"
                    f"Number of frequencies: {len(self.fluid_data['data'].get('original_frequencies', []))}"
                )
            else:
                self.quantum_data_text.setText("Invalid fluid data structure")

        except Exception as e:
            self.quantum_data_text.setText(f"Error importing fluid data: {str(e)}")
            print(f"Error importing fluid data: {str(e)}")
            traceback.print_exc()

    def setup_bloch_visualizer(self):
        """Initialize Bloch sphere visualizer."""
        try:
            print("Setting up Bloch visualizer...")

            # Ensure figure and canvas are initialized
            if not hasattr(self, "figure") or not hasattr(self, "canvas"):
                print("Figure or canvas not initialized. Creating new instances.")
                self.figure = Figure(figsize=(8, 8))
                self.canvas = FigureCanvas(self.figure)

            # Clear the figure
            self.figure.clear()

            # Initialize the Bloch visualizer with current figure
            self.bloch_visualizer = BlochSphereVisualizer(
                self.data_manager, self.figure, self.canvas
            )

            print("Bloch visualizer setup complete.")
            return True

        except Exception as e:
            print(f"Error setting up Bloch visualizer: {str(e)}")
            traceback.print_exc()
            return False

    def change_visualization_style(self, style):
        """Switch between visualization styles"""
        try:
            print(f"Changing visualization style to: {style}")

            # Stop any running animation first
            if self.timer.isActive():
                self.timer.stop()

            if style == "Bloch Sphere":
                # Update UI
                self.previous_viz_mode = self.viz_mode.currentText()
                self.viz_mode.blockSignals(True)
                self.viz_mode.clear()
                self.viz_mode.addItem("Quantum State")
                self.viz_mode.setCurrentText("Quantum State")
                self.viz_mode.blockSignals(False)

                # Clean up existing visualizer if it exists
                if (
                    hasattr(self, "bloch_visualizer")
                    and self.bloch_visualizer is not None
                ):
                    self.bloch_visualizer.cleanup()

                # Create new Bloch visualizer
                success = self.setup_bloch_visualizer()
                if not success:
                    print("Failed to setup Bloch visualizer")
                    return

                # Connect timer to Bloch visualization update
                try:
                    self.timer.timeout.disconnect()
                except:
                    pass
                self.timer.timeout.connect(self.update_bloch_visualization)
                self.timer.setInterval(100)  # Slower updates for Bloch sphere

            else:  # Switch back to Particle Simulation
                # Clean up Bloch visualizer if it exists
                if (
                    hasattr(self, "bloch_visualizer")
                    and self.bloch_visualizer is not None
                ):
                    self.bloch_visualizer.cleanup()
                    self.bloch_visualizer = None

                # Restore visualization modes
                self.viz_mode.blockSignals(True)
                self.viz_mode.clear()
                self.viz_mode.addItems(
                    [
                        "Velocity",
                        "Position",
                        "Energy",
                        "Quantum State",
                        "Spectral Lines",
                    ]
                )
                if hasattr(self, "previous_viz_mode"):
                    self.viz_mode.setCurrentText(self.previous_viz_mode)
                self.viz_mode.blockSignals(False)

                # Clear but don't replace the figure
                self.figure.clear()
                self.ax = self.figure.add_subplot(111, projection="3d")
                self.ax.set_facecolor(COLORS["background"])
                self._scatter = None

                # Update timer connection
                try:
                    self.timer.timeout.disconnect()
                except:
                    pass
                self.timer.timeout.connect(self.update_simulation)
                self.timer.setInterval(self.animation_speed)

            # Restart timer if it was previously running
            if self.toggle_button.text() == "Stop":
                self.timer.start()

        except Exception as e:
            print(f"Error changing visualization style: {str(e)}")
            traceback.print_exc()

    def update_bloch_visualization(self):
        """Update Bloch sphere visualization"""
        try:
            if hasattr(self, "bloch_visualizer") and self.bloch_visualizer:
                # Get the frame back from the visualization
                self._animation_frame = (
                    self.bloch_visualizer.update_animation(self._animation_frame) or 0
                )
                self._animation_frame += 1
        except Exception as e:
            print(f"Error updating Bloch visualization: {str(e)}")
            traceback.print_exc()

    def update_timestep(self, value):
        """Handle timestep slider value change"""
        try:
            # Update the label
            self.time_label.setText(f"Step: {value}")

            # Update Bloch sphere if active
            if (
                self.viz_style
                and self.viz_style.currentText() == "Bloch Sphere"
                and hasattr(self, "bloch_visualizer")
                and self.bloch_visualizer is not None
            ):

                # Temporarily stop animation if running
                was_active = self.timer.isActive()
                if was_active:
                    self.timer.stop()

                # Update visualization
                self.bloch_visualizer.update_for_timestep(value)

                # Restore animation if it was running
                if was_active:
                    self.timer.start()

        except Exception as e:
            print(f"Error in update_timestep: {e}")
            traceback.print_exc()

    def step_forward(self):
        """Step forward one timestep"""
        current = (
            self.time_slider.value()
        )  # Changed from timestep_slider to time_slider
        if current < self.time_slider.maximum():  # Changed here too
            self.time_slider.setValue(current + 1)

    def step_backward(self):
        """Step backward one timestep"""
        current = (
            self.time_slider.value()
        )  # Changed from timestep_slider to time_slider
        if current > 0:
            self.time_slider.setValue(current - 1)

    def __del__(self):
        self._cleanup_qt_objects()
        if hasattr(self, "timer") and self.timer is not None:
            self.timer.stop()
        if hasattr(self, "figure") and self.figure is not None:
            plt.close(self.figure)
