import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QSlider, QComboBox, QTextEdit, QTabWidget)
from PyQt6.QtCore import Qt, QTimer
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from core.calculations.particle.particle_simulator import ParticleSimulator
from storage.data_manager import QuantumDataManager
from ui.styles_ui.styles import *


class ParticleSimulationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_manager = QuantumDataManager()
        self.speed_label = None
        self._scatter = None
        self.colorbar = None
        self.toggle_button = None
        self.viz_mode = None
        self.import_quantum_btn = None
        self.speed_slider = None
        self.data_source = None
        self.animation_speed = 20
        self.simulator = ParticleSimulator(num_particles=1024)
        self.current_mode = "Energy"
        self.quantum_data_text = None

        self.import_fluid_btn = None
        self.fluid_data = None
        self.using_fluid_data = False

        # Setup matplotlib figure
        self.figure = Figure(figsize=(8, 8))
        self.figure.patch.set_facecolor(COLORS['background'])
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor(COLORS['background'])

        # Animation control
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.setInterval(self.animation_speed)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 10)  # Increased bottom margin
        layout.setSpacing(5)  # Slightly increased spacing between elements

        # Create Controls GroupBox
        controls_group = create_group_box("Particle Controls")
        controls_layout = QVBoxLayout()

        # First row of controls
        top_controls = QHBoxLayout()

        # Import and Data Source controls
        self.import_quantum_btn = create_button("Import Quantum Data")
        self.import_quantum_btn.setToolTip("Import data from Quantum Analysis")
        self.import_quantum_btn.clicked.connect(self.import_quantum_data)
        top_controls.addWidget(self.import_quantum_btn)

        # After self.import_quantum_btn
        self.import_fluid_btn = create_button("Import Fluid Data")
        self.import_fluid_btn.setToolTip("Import data from Fluid Dynamics")
        self.import_fluid_btn.clicked.connect(self.import_fluid_data)
        top_controls.addWidget(self.import_fluid_btn)

        # Update data source items to include Fluid Data

        # Data source dropdown with tooltip
        source_label = QLabel("Data Source:")
        source_label.setStyleSheet(base_style)
        top_controls.addWidget(source_label)

        self.data_source = QComboBox()
        self.data_source.addItems(["Particle Simulation", "Quantum Data", "Fluid Data"])
        self.data_source.setToolTip("Select data source for particle behavior")
        self.data_source.currentTextChanged.connect(self.change_data_source)
        top_controls.addWidget(self.data_source)

        # Reset and Start/Stop buttons
        reset_button = create_button("Reset")
        reset_button.setToolTip("Reset simulation to initial state")
        reset_button.clicked.connect(self.reset_simulation)
        top_controls.addWidget(reset_button)

        self.toggle_button = create_button("Start")
        self.toggle_button.setToolTip("Start/Stop simulation")
        self.toggle_button.clicked.connect(self.toggle_simulation)
        top_controls.addWidget(self.toggle_button)

        controls_layout.addLayout(top_controls)

        # Second row of controls
        viz_controls = QHBoxLayout()

        # Visualization mode
        viz_label = QLabel("Visualization Mode:")
        viz_label.setStyleSheet(base_style)
        self.viz_mode = QComboBox()
        self.viz_mode.addItems(["Velocity", "Position", "Energy", "Quantum State"])
        self.viz_mode.setToolTip("Select visualization type")
        self.viz_mode.currentTextChanged.connect(self.update_visualization)
        viz_controls.addWidget(viz_label)
        viz_controls.addWidget(self.viz_mode)

        # Speed control with labels
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Animation Speed:")
        speed_label.setStyleSheet(base_style)
        speed_layout.addWidget(speed_label)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(20, 500)
        self.speed_slider.setValue(100)
        self.speed_slider.setToolTip("Adjust simulation speed")
        self.speed_slider.valueChanged.connect(self.update_speed)
        self.speed_slider.setStyleSheet("color: white;")
        speed_layout.addWidget(self.speed_slider)

        self.speed_label = QLabel("100")
        self.speed_label.setStyleSheet(base_style)
        speed_layout.addWidget(self.speed_label)

        viz_controls.addLayout(speed_layout)
        controls_layout.addLayout(viz_controls)

        # Quantum Data Display
        quantum_group = create_group_box("Quantum Data")
        quantum_layout = QVBoxLayout()
        self.quantum_data_text = QTextEdit()
        self.quantum_data_text.setReadOnly(True)
        self.quantum_data_text.setStyleSheet(textedit_style)
        self.quantum_data_text.setMaximumHeight(100)
        quantum_layout.addWidget(self.quantum_data_text)
        quantum_group.setLayout(quantum_layout)
        controls_layout.addWidget(quantum_group)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Add canvas with background styling
        # canvas_group = create_group_box("Visualization")
        # canvas_layout = QVBoxLayout()
        #
        # self.canvas.setMinimumHeight(400)
        # self.canvas.setMaximumHeight(800)
        # canvas_layout.addWidget(self.canvas)
        # canvas_group.setLayout(canvas_layout)
        # layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)
        #
        # layout.addWidget(canvas_group)

        canvas_group = create_group_box("Visualization")
        canvas_layout = QVBoxLayout()

        # Add the canvas
        canvas_layout.addWidget(self.canvas, alignment=Qt.AlignmentFlag.AlignCenter)
        self.canvas.setMinimumHeight(400)
        self.canvas.setMaximumHeight(600)
        self.canvas.setContentsMargins(0, 5, 0, 0)
        canvas_layout.setContentsMargins(0, 0, 0, 20)
        canvas_layout.addStretch(1)

        canvas_group.setLayout(canvas_layout)
        layout.addWidget(canvas_group)

        # Style Sheets
        layout.addSpacing(10)
        self.setLayout(layout)

    def update_speed(self, value):
        """Update animation speed and label"""
        self.animation_speed = 3000 // value
        self.speed_label.setText(str(value))
        if self.timer.isActive():
            self.timer.setInterval(self.animation_speed)

    def import_fluid_data(self):
        """Import data from fluid dynamics tab"""
        try:
            # Get the fluid dynamics tab
            main_window = self.window()
            tabs = None
            for child in main_window.findChildren(QTabWidget):
                if "Fluid Dynamics" in [child.tabText(i) for i in range(child.count())]:
                    tabs = child
                    break

            if tabs:
                # Find the Fluid Dynamics tab
                for i in range(tabs.count()):
                    if tabs.tabText(i) == "Fluid Dynamics":
                        fluid_tab = tabs.widget(i)
                        # Get the current results
                        self.fluid_data = fluid_tab.get_current_results()
                        if self.fluid_data:
                            self.using_fluid_data = True
                            self.data_source.setCurrentText("Fluid Data")
                            self.quantum_data_text.setText("Fluid data imported successfully\n" +
                                                           f"Number of time points: {len(self.fluid_data['t'])}\n" +
                                                           f"Number of frequencies: "
                                                           f"{len(self.fluid_data['original_frequencies'])}")
                            return
                        break

            self.quantum_data_text.setText("No fluid data available")
        except Exception as e:
            self.quantum_data_text.setText(f"Error importing fluid data: {str(e)}")

    def import_quantum_data(self, quantum_data=None):
        try:
            # If quantum_data is directly provided, use it
            if quantum_data:
                self.simulator.apply_quantum_data(quantum_data)
                self._update_quantum_display(quantum_data)
                return

            # Otherwise try to get it from the Quantum Melody Analysis tab
            main_window = self.window()
            tabs = None
            for child in main_window.findChildren(QTabWidget):
                if "Quantum Melody Analysis" in [child.tabText(i) for i in
                                                 range(child.count())]:  # Changed from "Quantum Melody"
                    tabs = child
                    break

            if tabs:
                # Find the Quantum Melody Analysis tab
                for i in range(tabs.count()):
                    if tabs.tabText(i) == "Quantum Melody Analysis":  # Changed from "Quantum Melody"
                        quantum_tab = tabs.widget(i)
                        if hasattr(quantum_tab, 'last_results'):
                            quantum_data = quantum_tab.last_results

            if quantum_data:
                self.simulator.apply_quantum_data(quantum_data)
                self._update_quantum_display(quantum_data)
            else:
                self.quantum_data_text.setText(
                    "No quantum data available. Please run analysis in Quantum Melody Analysis tab first.")

        except Exception as e:
            self.quantum_data_text.setText(f"Error importing quantum data: {str(e)}")

    def change_data_source(self, source):
        """Handle data source changes"""
        if source == "Quantum Data" and not self.simulator.quantum_data:
            self.quantum_data_text.setText("No quantum data loaded. Please import data first.")
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
            self.colorbar.set_label(self.current_mode, color='white')
            self.colorbar.ax.yaxis.set_tick_params(color='white')
            self.colorbar.ax.yaxis.label.set_color('white')

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

    # This is where storage is stored
    def update_simulation(self):
        """Update particle visualization"""
        try:
            positions, velocities, accelerations = self.simulator.update(dt=0.01)

            # Store results in data manager
            simulation_results = {
                'positions': positions.tolist(),
                'velocities': velocities.tolist(),
                'accelerations': accelerations.tolist(),
                'mode': self.current_mode,
                'time': self.timer.interval()
            }
            self.data_manager.update_particle_results(simulation_results)

            if not self._scatter:
                # Initial setup only once
                self.figure.clear()
                self.ax = self.figure.add_subplot(111, projection='3d')
                self.ax.set_facecolor(COLORS['background'])
                self.ax.set_xlim(-8, 8)
                self.ax.set_ylim(-8, 8)
                self.ax.set_zlim(-8, 8)
                self.ax.grid(True)
                self.ax.set_position([0.1, 0.1, 0.8, 0.8])  # [left, bottom, width, height]

                # Style setup
                self.ax.xaxis.label.set_color('white')
                self.ax.yaxis.label.set_color('white')
                self.ax.zaxis.label.set_color('white')
                self.ax.tick_params(colors='white')

                # Create scatter plot with initial colors
                colors = self._get_colors(positions, velocities)
                self._scatter = self.ax.scatter(
                    positions[:, 0], positions[:, 1], positions[:, 2],
                    c=colors,
                    cmap='viridis',
                    alpha=0.6
                )

            else:
                # Update positions and colors
                self._scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
                colors = self._get_colors(positions, velocities)
                self._scatter.set_array(colors)

            # Always update colorbar
            if self.colorbar is not None:
                self.colorbar.remove()
            self.colorbar = self.figure.colorbar(self._scatter)
            self.colorbar.set_label(self.current_mode, color='white')
            # Fix all colorbar text colors
            self.colorbar.ax.yaxis.set_tick_params(colors='white')
            self.colorbar.ax.yaxis.label.set_color('white')
            # Add this line to color the numbers
            plt.setp(plt.getp(self.colorbar.ax.axes, 'yticklabels'), color='white')

            # Draw only when needed
            if self.timer.isActive():
                self.canvas.draw_idle()

        except Exception as e:
            print(f"Error updating simulation: {str(e)}")

    def _update_quantum_display(self, quantum_data):
        """Helper method to update the quantum data display"""
        try:
            display_text = []
            if 'quantum_metrics' in quantum_data:
                metrics = quantum_data['quantum_metrics']
                display_text.append(f"Purity: {metrics['purity']:.3f}")
                display_text.append(f"Fidelity: {metrics['fidelity']:.3f}")
            if 'frequencies' in quantum_data:
                display_text.append(f"Number of frequencies: {len(quantum_data['frequencies'])}")
            if 'musical_systems' in quantum_data:
                display_text.append("\nMusical Systems:")
                for system, notes in quantum_data['musical_systems'].items():
                    display_text.append(f"{system}: {', '.join(notes)}")

            self.quantum_data_text.setText('\n'.join(display_text))
            self.quantum_data_text.append("\nQuantum data applied to simulation")
            self.data_source.setCurrentText("Quantum Data")
        except Exception as e:
            self.quantum_data_text.setText(f"Error updating quantum display: {str(e)}")

    def _get_colors(self, positions, velocities):
        """Get colors based on current visualization mode"""
        if self.current_mode == "Velocity":
            return np.linalg.norm(velocities, axis=1)
        elif self.current_mode == "Position":
            return np.linalg.norm(positions, axis=1)
        elif self.current_mode == "Energy":
            return 0.5 * np.sum(velocities ** 2, axis=1)
        else:  # Quantum State
            if self.simulator.quantum_data:
                # Get statevector and ensure it matches particle count
                statevector = self.simulator.quantum_data.get('statevector', np.zeros(len(positions)))
                # Tile the statevector to match particle count
                if len(statevector) != len(positions):
                    statevector = np.tile(statevector, (len(positions) + len(statevector) - 1) // len(statevector))[
                                  :len(positions)]
                # Scale the values to be visible (they might be very small)
                values = np.abs(statevector)
                if values.max() > 0:  # Normalize only if we have non-zero values
                    values = values / values.max()
                return values
            return np.zeros(len(positions))  # Default if no quantum data
