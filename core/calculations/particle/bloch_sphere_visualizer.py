import numpy as np
import traceback
from PyQt6.QtCore import QMutex, QMutexLocker, QTimer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from data.backend_data_management.data_manager import QuantumDataManager

import numpy as np
from PyQt6.QtCore import QMutex, QMutexLocker, QTimer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

import numpy as np
from PyQt6.QtCore import QMutex, QMutexLocker, QTimer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

import numpy as np
import traceback
from PyQt6.QtCore import QMutex, QMutexLocker, QTimer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

import numpy as np
import traceback
from typing import Dict, Any, Tuple


class BlochSphereVisualizer:
    def __init__(self, data_manager, figure, canvas):
        # Prevent multiple initializations
        if hasattr(self, "_initialized"):
            return
        self._initialized = True

        # Strong references to prevent garbage collection
        self._figure_ref = figure
        self._canvas_ref = canvas

        # Initialize cleanup flag
        self._is_cleanup = False
        self.cleanup_count = 0

        # Initialize mutexes and other components
        self._canvas_mutex = QMutex()
        self._data_mutex = QMutex()

        # Store references
        self.data_manager = data_manager
        self.figure = figure
        self._canvas = canvas

        # Prevent multiple cleanup attempts
        self._cleanup_in_progress = False

        # Animation state
        self.animation_frame = 0
        self.state_history = []
        self.current_timestep = 0
        self.max_history = 50

        print("Initializing BlochSphereVisualizer")
        self.setup_bloch_sphere()

    def setup_bloch_sphere(self):
        """Initial setup of Bloch sphere"""
        print("Setting up Bloch sphere visualization...")

        try:
            # Clear any existing plots
            self.figure.clear()

            # Create sphere with more points for smoother appearance
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))

            # Create 3D axis
            ax = self.figure.add_subplot(111, projection="3d")

            # Plot wireframe sphere
            ax.plot_wireframe(x, y, z, color="gray", alpha=0.2, linewidth=0.5)

            # Set sphere properties
            ax.set_xlabel("X", color="white")
            ax.set_ylabel("Y", color="white")
            ax.set_zlabel("Z", color="white")

            # Set limits to ensure full sphere is visible
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])
            ax.set_zlim([-1.2, 1.2])

            # Add basis state labels
            ax.text(0, 0, 1.1, "|0⟩", color="white", ha="center", va="bottom")
            ax.text(0, 0, -1.1, "|1⟩", color="white", ha="center", va="top")

            # Set background color to specified hex color
            background_color = "#1f2f4a"
            ax.set_facecolor(background_color)
            self.figure.patch.set_facecolor(background_color)

            # Customize axis
            ax.grid(False)
            ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

            # Add a spot for the quantum state
            self.quantum_point = ax.scatter(
                [], [], [], c="red", s=100, label="Quantum State"
            )

            # Add trajectory for state evolution
            self.trajectory = ax.plot(
                [], [], [], "g-", alpha=0.7, linewidth=2, label="State History"
            )[0]

            # Initialize trajectory history
            self.trajectory_history = {"x": [], "y": [], "z": []}

            # Add legend
            ax.legend(loc="upper right")

            # Ensure canvas is drawn
            if self._canvas:
                try:
                    self._canvas.draw()
                    self._canvas.flush_events()
                except Exception as canvas_error:
                    print(f"Canvas draw error: {canvas_error}")

            print("Bloch sphere setup complete")

        except Exception as e:
            print(f"Error in setup_bloch_sphere: {str(e)}")
            traceback.print_exc()

    def extract_bloch_coordinates(
        quantum_data: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """
        Advanced method to extract Bloch sphere coordinates from complex quantum data

        Args:
            quantum_data (dict): Comprehensive quantum system data

        Returns:
            tuple: (x, y, z) coordinates on Bloch sphere
        """
        try:
            # Extensive logging of input data
            print("DEBUG - Bloch Coordinate Extraction:")
            print(f"Input data keys: {quantum_data.keys()}")

            # Extract statevector with multiple fallback strategies
            statevector = None
            statevector_keys = ["statevector", "quantum_state", "state"]
            for key in statevector_keys:
                statevector = quantum_data.get(key)
                if statevector is not None:
                    break

            # If no statevector, try creating one from frequencies
            if statevector is None:
                frequencies = quantum_data.get("quantum_frequencies", [])
                if frequencies:
                    print(
                        f"DEBUG: Using frequencies to construct statevector. Frequencies: {frequencies}"
                    )
                    # Normalize frequencies to create a pseudo-statevector
                    freq_array = np.array(frequencies)
                    statevector = freq_array / np.linalg.norm(freq_array)

            # Validate statevector
            if statevector is None:
                print("ERROR: No suitable statevector found")
                return 0, 0, 1  # Default to |0⟩ state

            # Convert to numpy array if not already
            statevector = np.asarray(statevector, dtype=complex)

            print(f"DEBUG: Statevector shape: {statevector.shape}")
            print(f"DEBUG: Statevector magnitudes: {np.abs(statevector)}")

            # Handle different dimensionality
            if statevector.ndim > 1:
                statevector = statevector.flatten()

            # Strategy for multi-dimensional state
            if len(statevector) > 2:
                # Take top 2 components by magnitude
                top_indices = np.argsort(np.abs(statevector))[-2:]
                reduced_state = statevector[top_indices]

                print(f"DEBUG: Reduced state: {reduced_state}")
                print(f"DEBUG: Reduced state magnitudes: {np.abs(reduced_state)}")
            else:
                reduced_state = statevector

            # Normalize reduced state
            reduced_state = reduced_state / np.linalg.norm(reduced_state)

            # Compute Bloch sphere coordinates
            if len(reduced_state) >= 2:
                x = 2 * np.real(reduced_state[0] * np.conj(reduced_state[1]))
                y = 2 * np.imag(reduced_state[0] * np.conj(reduced_state[1]))
                z = np.abs(reduced_state[0]) ** 2 - np.abs(reduced_state[1]) ** 2

                print(f"DEBUG: Computed Bloch coordinates: (x={x}, y={y}, z={z})")
                return x, y, z

            # Fallback for unexpected state
            print("WARNING: Unable to compute full Bloch coordinates")
            return 0, 0, 1

        except Exception as e:
            print(f"CRITICAL ERROR in Bloch coordinate extraction: {e}")
            traceback.print_exc()
            return 0, 0, 1

    def debug_quantum_data(quantum_data: Dict[str, Any]):
        """
        Comprehensive debugging of quantum data

        Args:
            quantum_data (dict): Quantum system data to inspect
        """
        print("\n--- QUANTUM DATA DEBUGGING ---")
        print("Data Keys:", quantum_data.keys())

        # Check and print specific keys
        debug_keys = [
            "statevector",
            "quantum_frequencies",
            "density_matrix",
            "purity",
            "fidelity",
        ]

        for key in debug_keys:
            if key in quantum_data:
                value = quantum_data[key]
                print(f"\n{key.upper()}:")

                # Different handling based on type
                if isinstance(value, (list, np.ndarray)):
                    print(f"  Length: {len(value)}")
                    print(f"  First few elements: {value[:5]}")
                    print(f"  Magnitude of first elements: {np.abs(value[:5])}")
                elif hasattr(value, "shape"):
                    print(f"  Shape: {value.shape}")
                else:
                    print(f"  Value: {value}")

        print("\n--- END QUANTUM DATA DEBUGGING ---\n")

    def update_animation(self, frame):
        """Update animation for the given frame"""
        try:
            success = self.update_for_timestep(frame)
            if success:
                self.animation_frame = frame
            return frame
        except Exception as e:
            print(f"Error in update_animation: {str(e)}")
            traceback.print_exc()
            return 0

    def show_figure(self, fig):
        """
        Create a new figure manager and reassign the canvas
        See https://github.com/Qiskit/qiskit-terra/issues/1682
        """
        try:
            # Create a new figure
            new_fig = plt.figure()
            # Get the new figure's canvas manager
            new_mngr = new_fig.canvas.manager
            # Set the canvas of the new manager to the original figure
            new_mngr.canvas.figure = fig
            # Set the figure's canvas to the new manager's canvas
            fig.set_canvas(new_mngr.canvas)

            # Explicitly draw the figure
            fig.canvas.draw()

            print("DEBUG: Figure shown successfully")
            return new_fig
        except Exception as e:
            print(f"ERROR in show_figure: {e}")
            return None

    def maintain_canvas_reference(self):
        """
        Ensure canvas references are maintained and valid
        """
        try:
            # Check if canvas exists and is valid
            if (
                self._canvas is None
                or not hasattr(self._canvas, "figure")
                or self._canvas.figure is None
            ):

                # Try to restore from strong references
                if hasattr(self, "_figure_ref") and self._figure_ref is not None:
                    self.figure = self._figure_ref
                if hasattr(self, "_canvas_ref") and self._canvas_ref is not None:
                    self._canvas = self._canvas_ref

                # Log restoration attempt
                print("Attempting to restore canvas reference")

            # Validate restored references
            if self._canvas is None:
                print("ERROR: Unable to maintain canvas reference")
                return False

            return True

        except Exception as e:
            print(f"Error maintaining canvas reference: {e}")
            return False

    def update_for_timestep(self, timestep):
        """
        Update Bloch sphere visualization for a specific timestep

        Args:
            timestep (int): Current animation timestep

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            print(f"DEBUG: Attempting to update Bloch sphere for timestep {timestep}")
            if not isinstance(timestep, int):
                print(f"Invalid timestep type: {type(timestep)}")
                return False

            print(f"DEBUG: Attempting to update Bloch sphere for timestep {timestep}")

            # Check if canvas and figure still exist
            if (
                not hasattr(self, "_canvas")
                or self._canvas is None
                or not hasattr(self, "figure")
                or self.figure is None
            ):
                print("ERROR: Canvas or figure has been deleted")
                return False

            # Maintain canvas reference
            if not self.maintain_canvas_reference():
                print("ERROR: Lost canvas reference")
                return False

            # Single mutex lock for quantum data processing
            with QMutexLocker(self._data_mutex):
                # Fetch quantum results
                quantum_results = self.data_manager.quantum_results
                print(f"DEBUG: Total quantum results: {len(quantum_results)}")

                if not quantum_results:
                    print("ERROR: No quantum results found!")
                    return False

                total_steps = len(quantum_results)
                result_idx = min(timestep % total_steps, total_steps - 1)

                # Ensure we have valid data
                latest_result = quantum_results[result_idx]
                quantum_data = latest_result.get("data", latest_result)

                print("DEBUG - Quantum Data Keys:", quantum_data.keys())

                # Multiple strategies to find statevector
                statevector_keys = [
                    "statevector",
                    "quantum_state",
                    "state",
                    "amplitudes",
                ]

                statevector = None
                for key in statevector_keys:
                    statevector = quantum_data.get(key)
                    if statevector is not None:
                        print(f"DEBUG: Found statevector using key '{key}'")
                        break

                # If no statevector found, try frequencies
                if statevector is None:
                    frequencies = quantum_data.get("quantum_frequencies", [])
                    if frequencies:
                        print("DEBUG: Creating statevector from frequencies")
                        statevector = np.sqrt(np.abs(frequencies) / np.sum(frequencies))

                # Validate statevector
                if statevector is None:
                    print("ERROR: No statevector found in quantum data")
                    return False

                # Convert to numpy array
                statevector = np.asarray(statevector, dtype=complex)

                # Print diagnostic information
                print(f"DEBUG: Statevector shape: {statevector.shape}")
                print(f"DEBUG: Statevector first few elements: {statevector[:5]}")
                print(f"DEBUG: Statevector magnitudes: {np.abs(statevector)}")

                # Handle multi-dimensional statevector
                if statevector.ndim > 1:
                    statevector = statevector.flatten()

                # Strategy for multi-dimensional state
                if len(statevector) > 2:
                    # Take top 2 components by magnitude
                    top_indices = np.argsort(np.abs(statevector))[-2:]
                    reduced_state = statevector[top_indices]

                    print(f"DEBUG: Reduced state indices: {top_indices}")
                    print(f"DEBUG: Reduced state: {reduced_state}")
                    print(f"DEBUG: Reduced state magnitudes: {np.abs(reduced_state)}")
                else:
                    reduced_state = statevector

                # Normalize reduced state
                reduced_state = reduced_state / np.linalg.norm(reduced_state)

                # Compute Bloch sphere coordinates
                x = 2 * np.real(reduced_state[0] * np.conj(reduced_state[1]))
                y = 2 * np.imag(reduced_state[0] * np.conj(reduced_state[1]))
                z = np.abs(reduced_state[0]) ** 2 - np.abs(reduced_state[1]) ** 2

                print(f"DEBUG: Computed Bloch coordinates: (x={x}, y={y}, z={z})")

            with QMutexLocker(self._canvas_mutex):
                # Check canvas is still valid
                if not hasattr(self, "_canvas") or self._canvas is None:
                    print("ERROR: Canvas has been deleted during visualization")
                    return False

                # Define background color
                background_color = "#1f2f4a"

                try:
                    # Recreate the Bloch sphere
                    if not hasattr(self, "figure") or self.figure is None:
                        print("ERROR: Figure has been deleted")
                        return False

                    self.figure.clear()
                    ax = self.figure.add_subplot(111, projection="3d")

                    # Set background color
                    ax.set_facecolor(background_color)
                    self.figure.patch.set_facecolor(background_color)

                    # Create sphere wireframe
                    u = np.linspace(0, 2 * np.pi, 100)
                    v = np.linspace(0, np.pi, 100)
                    sphere_x = np.outer(np.cos(u), np.sin(v))
                    sphere_y = np.outer(np.sin(u), np.sin(v))
                    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))

                    # Plot wireframe sphere
                    ax.plot_wireframe(
                        sphere_x,
                        sphere_y,
                        sphere_z,
                        color="gray",
                        alpha=0.2,
                        linewidth=0.5,
                    )

                    # Set sphere properties
                    ax.set_xlabel("X", color="white")
                    ax.set_ylabel("Y", color="white")
                    ax.set_zlabel("Z", color="white")

                    # Set limits to ensure full sphere is visible
                    ax.set_xlim([-1.2, 1.2])
                    ax.set_ylim([-1.2, 1.2])
                    ax.set_zlim([-1.2, 1.2])

                    # Add basis state labels
                    ax.text(0, 0, 1.1, "|0⟩", color="white", ha="center", va="bottom")
                    ax.text(0, 0, -1.1, "|1⟩", color="white", ha="center", va="top")

                    # Reset or initialize trajectory history if it doesn't exist
                    if not hasattr(self, "trajectory_history"):
                        self.trajectory_history = {"x": [], "y": [], "z": []}

                    # Add current point to trajectory history
                    self.trajectory_history["x"].append(x)
                    self.trajectory_history["y"].append(y)
                    self.trajectory_history["z"].append(z)

                    # Limit trajectory history
                    max_trail = 50
                    if len(self.trajectory_history["x"]) > max_trail:
                        self.trajectory_history["x"] = self.trajectory_history["x"][
                            -max_trail:
                        ]
                        self.trajectory_history["y"] = self.trajectory_history["y"][
                            -max_trail:
                        ]
                        self.trajectory_history["z"] = self.trajectory_history["z"][
                            -max_trail:
                        ]

                    # Plot quantum state point (red)
                    ax.scatter(x, y, z, color="red", s=100, label="Quantum State")

                    # Plot trajectory (green)
                    ax.plot(
                        self.trajectory_history["x"],
                        self.trajectory_history["y"],
                        self.trajectory_history["z"],
                        color="green",
                        linewidth=2,
                        alpha=0.7,
                        label="State Trajectory",
                    )

                    # Add legend and customize colors
                    ax.legend(
                        loc="upper right",
                        facecolor="#162030",
                        edgecolor="white",
                        framealpha=0.7,
                    )
                    plt.setp(ax.get_legend().get_texts(), color="white")

                    # Customize axis
                    ax.grid(False)
                    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

                    # Force canvas update
                    try:
                        if not hasattr(self, "_canvas") or self._canvas is None:
                            print("ERROR: Canvas has been deleted before drawing")
                            return False

                        # Use draw_idle to prevent blocking
                        from PyQt6.QtCore import QTimer

                        QTimer.singleShot(0, self._canvas.draw)
                    except Exception as draw_error:
                        print(f"Error drawing canvas: {draw_error}")
                        return False

                    return True

                except Exception as viz_error:
                    print(f"CRITICAL ERROR in Bloch sphere visualization: {viz_error}")
                    traceback.print_exc()
                    return False

                frequencies = quantum_data.get("quantum_frequencies", [])
                # Prepare state information text
                state_info_text = f"Timestep: {timestep}\n"

                if frequencies:
                    # Get top frequencies
                    top_freq_indices = np.argsort(frequencies)[-3:][::-1]
                    top_frequencies = [frequencies[i] for i in top_freq_indices]

                    state_info_text += "Top Frequencies:\n"
                    for i, freq in enumerate(top_frequencies, 1):
                        state_info_text += f"{i}. {freq:.2f} Hz\n"

                # Add state vector information
                state_info_text += "\nState Vector:\n"
                state_info_text += f"x: {x:.4f}\n"
                state_info_text += f"y: {y:.4f}\n"
                state_info_text += f"z: {z:.4f}\n"

                # Add purity and fidelity if available
                purity = quantum_data.get("purity", 0)
                fidelity = quantum_data.get("fidelity", 0)
                state_info_text += f"\nPurity: {purity:.4f}\n"
                state_info_text += f"Fidelity: {fidelity:.4f}"

                # Add text to the figure
                self.figure.text(
                    0.02,
                    0.02,  # Lower left corner
                    state_info_text,
                    color="white",
                    fontsize=8,
                    transform=self.figure.transFigure,
                    verticalalignment="bottom",
                    bbox=dict(facecolor="#162030", alpha=0.7, edgecolor="none"),
                )

                return True

        except Exception as e:
            print(f"FATAL ERROR in update_for_timestep: {e}")
            traceback.print_exc()
            return False

    def get_latest_quantum_state(self):
        """
        Retrieve the latest quantum state from the data manager

        Returns:
            dict or None: The most recent quantum data
        """
        try:
            # Assuming data_manager has a method to get latest results
            quantum_results = self.data_manager.quantum_results

            if not quantum_results:
                print("No quantum results found")
                return None

            # Get the most recent result
            latest_result = quantum_results[-1]

            # Print debug information
            print("DEBUG - Available keys in latest quantum result:")
            print(
                latest_result.keys()
                if isinstance(latest_result, dict)
                else "Not a dictionary"
            )

            # Try to extract statevector from different possible locations
            statevector_keys = [
                "statevector",
                "quantum_state",
                "state",
                "density_matrix",
            ]

            for key in statevector_keys:
                if key in latest_result:
                    statevector = latest_result[key]
                    print(f"DEBUG: Found statevector using key '{key}'")
                    print(f"DEBUG: Statevector type: {type(statevector)}")
                    print(f"DEBUG: Statevector length: {len(statevector)}")
                    return statevector

            print("ERROR: No statevector found in quantum results")
            return None

        except Exception as e:
            print(f"Error retrieving quantum state: {e}")
            return None

    def _deferred_draw(self):
        """Handle deferred canvas updates"""
        with QMutexLocker(self._canvas_mutex):
            try:
                if self._canvas:
                    self._canvas.draw()
            except RuntimeError as e:
                print(f"Deferred draw error: {e}")
            self._pending_update = False

    def cleanup(self):
        """
        Soft cleanup that preserves canvas and figure
        """
        if getattr(self, "_cleanup_in_progress", False):
            return

        try:
            self._cleanup_in_progress = True

            # Stop any timers
            if hasattr(self, "_update_timer"):
                self._update_timer.stop()

            # Clear figure content instead of deleting
            if self.figure:
                self.figure.clear()

            # Prevent canvas deletion
            if self._canvas:
                try:
                    self._canvas.draw_idle()
                except Exception as draw_error:
                    print(f"Error in canvas draw_idle: {draw_error}")

        except Exception as e:
            print(f"Error in BlochSphereVisualizer cleanup: {e}")

        finally:
            self._cleanup_in_progress = False
            self._is_cleanup = True
