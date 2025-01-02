from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QApplication,
    QMessageBox,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal

from storage.data_manager import QuantumDataManager
from ui.features.ai_ui.ai_analysis_screen import AIAnalysisScreen
from ui.features.quantum_ui.tab_ui.fluid_dynamics_tab import FluidAnalysisTab
from ui.features.quantum_ui.tab_ui.melody_analysis_tab import MelodyAnalysisTab
from ui.features.quantum_ui.tab_ui.particle_simulation_tab import ParticleSimulationTab
from ui.features.quantum_ui.tab_ui.qec_analysis_tab import QECAnalysisTab
from ui.features.quantum_ui.tab_ui.circuit_tab import QuantumMelodyAnalysisTab
from ui.features.quantum_ui.tab_ui.quantum_analysis_tab import QuantumAnalysisTab


class HarmonicsScreen(QWidget):
    analysis_complete = pyqtSignal(dict)

    def __init__(self, quantum_tabs=None):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.data_manager = QuantumDataManager()

        if quantum_tabs is None:
            raise ValueError("quantum_tabs cannot be None")

        self.quantum_tabs = quantum_tabs
        self.tabs = None
        self.setup_ui()
        self.connect_components()

        # Allow resizing
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def setup_ui(self):
        try:
            # Create and set up tabs
            self.tabs = QTabWidget()  # Store tabs as instance variable
            if self.quantum_tabs:
                for title, tab_class in self.quantum_tabs.items():
                    print(f"Adding tab: {title}")
                    tab = tab_class()
                    self.tabs.addTab(tab, title)
                    QApplication.processEvents()

            # Make tabs scrollable
            self.tabs.elideMode()
            # Style the tabs
            self.tabs.setStyleSheet(
                """
                   QTabWidget::pane {
                       border: 1px solid #3d405e;
                       background: #1f2f4a;
                       border-radius: 5px;
                   }
                   QTabBar::tab {
                       background: #3d405e;
                       color: white;
                       padding: 10px 20px;
                       border-top-left-radius: 5px;
                       border-top-right-radius: 5px;
                       margin-right: 2px;
                       font-family: Arial;
                       font-size: 14px;
                   }
                   QTabBar::tab:selected {
                       background: #1976D2;
                   }
                   QTabBar::tab:hover {
                       background: #1E88E5;
                   }
               """
            )

            # Tab Descriptors
            # 0 Quantum Analysis
            # 1 Melody Circuit
            # 2 Melody Analysis
            # 3 Note Table
            # 4 Fluid Dynamics
            # 5 QEC
            # 6 Simulator
            self.tabs.setTabToolTip(0, "Import audio for quantum analysis")
            self.tabs.setTabToolTip(1, "Input melody and export to simulator & QEC")
            self.tabs.setTabToolTip(2, "Input melody and analyze quantum effects")
            self.tabs.setTabToolTip(3, "View notes available for melody")
            self.tabs.setTabToolTip(
                4, "Analyze quantum and fluid dynamics and export to simulator"
            )
            self.tabs.setTabToolTip(5, "View melody with Quantum Error Correction")

            self.tabs.show()

            # Create main layout and add tabs
            main_layout = QVBoxLayout(self)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.addWidget(self.tabs)

        except Exception as e:
            print(f"Error setting up Harmonics: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to load Quantum Harmonics: {str(e)}"
            )

    def open_ai_analysis(self):
        # Create and show AI Analysis screen
        ai_screen = AIAnalysisScreen()

        # If you're using a tab-based interface, you might do:
        if hasattr(self, "quantum_tabs"):
            self.quantum_tabs.addTab(ai_screen, "AI Analysis")
            self.quantum_tabs.setCurrentWidget(ai_screen)
        else:
            # Alternative: replace current screen
            parent_layout = self.parent().layout()
            # Clear existing layout
            while parent_layout.count():
                child = parent_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            # Add AI screen
            parent_layout.addWidget(ai_screen)

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
