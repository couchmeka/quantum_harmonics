from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QApplication,
    QMessageBox,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal

from data.backend_data_management.data_manager import QuantumDataManager
from ui.features.ai_ui.ai_analysis_screen import AIAnalysisScreen
from ui.features.quantum_ui.tab_ui.fluid_dynamics_tab import FluidAnalysisTab
from ui.features.quantum_ui.tab_ui.melody_analysis_tab import MelodyAnalysisTab
from ui.features.quantum_ui.tab_ui.particle_simulation_tab import ParticleSimulationTab
from ui.features.quantum_ui.tab_ui.qec_analysis_tab import QECAnalysisTab
from ui.features.quantum_ui.tab_ui.circuit_tab import QuantumMelodyAnalysisTab
from ui.features.quantum_ui.tab_ui.quantum_analysis_tab import QuantumAnalysisTab


class HarmonicsScreen(QWidget):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, quantum_tabs=None, data_manager=None):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # Fix the data_manager initialization
        self.data_manager = (
            data_manager if data_manager is not None else QuantumDataManager()
        )

        if quantum_tabs is None:
            raise ValueError("quantum_tabs cannot be None")

        self.quantum_tabs = quantum_tabs
        self.tabs = None

        # Set window minimum size
        self.setMinimumSize(800, 600)

        # Configure the widget to expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Set background color for the main widget
        self.setStyleSheet(
            """
            HarmonicsScreen {
                background-color: #1f2f4a;
            }
        """
        )

        self.setup_ui()
        self.connect_components()

    def setup_ui(self):
        try:
            # Create main layout with zero margins to extend to edges
            main_layout = QVBoxLayout(self)
            # left, top, right, bottom
            main_layout.setContentsMargins(10, 20, 10, 10)
            main_layout.setSpacing(10)

            # Create and set up tabs
            self.tabs = QTabWidget()
            self.tabs.setDocumentMode(True)  # This makes tabs look more modern

            # Set minimum size for the tab widget
            self.tabs.setMinimumSize(800, 600)

            if self.quantum_tabs:
                for title, tab_class in self.quantum_tabs.items():
                    print(f"Adding tab: {title}")
                    try:
                        tab = tab_class(data_manager=self.data_manager)
                    except TypeError:
                        tab = tab_class()
                    self.tabs.addTab(tab, title)
                    QApplication.processEvents()

                    # Set accessible name for all tab contents
                    for i in range(self.tabs.count()):
                        self.tabs.widget(i).setAccessibleName("tabContent")

            # Add AI Analysis and Simulator tabs
            ai_analysis_screen = AIAnalysisScreen(data_manager=self.data_manager)
            self.tabs.addTab(ai_analysis_screen, "AI Analysis")
            particle_simulation_tab = ParticleSimulationTab(
                data_manager=self.data_manager
            )
            self.tabs.addTab(particle_simulation_tab, "Simulator")

            # Apply styling to the tab widget and its components
            # Style just the tabs and background
            self.setStyleSheet(
                """
                HarmonicsScreen {
                    background-color: #1f2f4a;
                }
                QTabWidget::pane { 
                    border: 1px solid #3d405e;
                    background-color: #1f2f4a;
                }
                QTabBar::tab {
                    background: #3d405e;
                    color: white;
                    padding: 8px 16px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                    margin-right: 1px;
                    font-family: Arial;
                    font-size: 13px;
                    min-width: 80px;
                }
                QTabBar::tab:selected {
                    background: #1976D2;
                }
                QTabBar::tab:hover {
                    background: #1E88E5;
                }
                QWidget[accessibleName="tabContent"] {
                    background-color: #1f2f4a;
                }
            """
            )

            # Center the tabs
            tabBar = self.tabs.tabBar()
            tabBar.setExpanding(True)
            tabBar.setDrawBase(False)

            # Add tabs to main layout
            main_layout.addWidget(self.tabs)

        except Exception as e:
            print(f"Error setting up Harmonics: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to load Quantum Harmonics: {str(e)}"
            )

    # In HarmonicsScreen class, update the connect_components method
    def connect_components(self):
        """Connect signals between tabs"""
        try:
            for i in range(self.tabs.count()):
                tab = self.tabs.widget(i)

                # Connect each tab type to update the data manager
                if isinstance(tab, FluidAnalysisTab):
                    tab.analysis_complete.connect(
                        lambda results: self.data_manager.update_fluid_results(results)
                    )
                    print("Connected Fluid tab to Data Manager")

                elif isinstance(tab, QECAnalysisTab):
                    tab.analysis_complete.connect(
                        lambda results: self.data_manager.update_qec_results(results)
                    )
                    print("Connected QEC tab to Data Manager")

                elif isinstance(tab, QuantumMelodyAnalysisTab):
                    tab.analysis_complete.connect(
                        lambda results: self.data_manager.update_quantum_results(
                            results
                        )
                    )
                    print("Connected Quantum tab to Data Manager")

                elif isinstance(tab, QuantumAnalysisTab):
                    tab.analysis_complete.connect(
                        lambda results: self.data_manager.update_quantum_results(
                            results
                        )
                    )
                    print("Connected Quantum Analysis to Data Manager")

                elif isinstance(tab, MelodyAnalysisTab):
                    tab.analysis_complete.connect(
                        lambda results: self.data_manager.update_melody_results(results)
                    )
                    print("Connected Melody Analysis to Data Manager")

            # Find specific tabs for direct connections
            fluid_tab = self.findChild(FluidAnalysisTab)
            qec_tab = self.findChild(QECAnalysisTab)
            quantum_tab = self.findChild(QuantumMelodyAnalysisTab)
            particle_tab = self.findChild(ParticleSimulationTab)

            # Connect tab-to-tab data flow
            if fluid_tab and qec_tab:
                fluid_tab.analysis_complete.connect(qec_tab.import_fluid_data)
                print("Connected Fluid Analysis to QEC")

            if quantum_tab and qec_tab:
                quantum_tab.analysis_complete.connect(qec_tab.import_quantum_data)
                print("Connected Quantum Analysis to QEC")

            if quantum_tab and particle_tab:
                quantum_tab.analysis_complete.connect(particle_tab.import_quantum_data)
                print("Connected Quantum Analysis to Particle Simulation")

        except Exception as e:
            print(f"Error connecting components: {str(e)}")

    def update_results(self, results):
        # When analysis is complete
        print("Updating quantum results:", results)
        self.data_manager.update_quantum_results(results)
        self.analysis_complete.emit(results)
