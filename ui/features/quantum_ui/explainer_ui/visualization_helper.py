# visualization_explanations.py

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel,
                             QGroupBox, QScrollArea)
from PyQt6.QtCore import Qt


class VisualizationExplanations(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Visualization Guide", parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.setStyleSheet("""
            QGroupBox {
                color: white;
                font-size: 14px;
                border: 1px solid #3d405e;
                border-radius: 5px;
                padding: 15px;
                background-color: rgba(61, 64, 94, 0.3);
            }
            QLabel {
                color: white;
                padding: 10px;
                background-color: rgba(31, 47, 74, 0.5);
                border-radius: 5px;
                margin: 5px;
            }
        """)

        explanations = {
            "Quantum Circuit": """
                Visual representation of the quantum operations:
                - Shows quantum gates applied to analyze audio
                - Displays how frequencies are encoded into quantum states
                - Illustrates quantum transformations and measurements
                - Demonstrates the quantum-classical interface
            """,
            "Quantum State Surface": """
                3D visualization of the quantum state space:
                - Peaks show high-probability quantum states
                - Valleys indicate low-probability states
                - Surface patterns reveal harmonic relationships
                - Color gradient indicates interaction strength
                - Symmetries may reveal Pythagorean ratios
            """,
            "State Probabilities": """
                Displays quantum measurement probabilities:
                - Each bar represents a distinct quantum state
                - Height shows probability of measuring that state
                - States are labeled in quantum notation
                - Distribution reveals harmonic patterns
                - Clustering indicates related frequencies
            """,
            "Quantum Quality Metrics": """
                Key metrics for quantum state analysis:
                - Quantum Purity: How "clean" the quantum state is
                  (Higher is better, indicates clearer harmonic separation)
                - Quantum Fidelity: How well preserved the state is
                  (Higher is better, shows less noise interference)
            """
        }

        for title, text in explanations.items():
            group = QGroupBox(title)
            group.setStyleSheet("""
                QGroupBox {
                    color: white;
                    border: 1px solid #1976D2;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
            """)

            label = QLabel(text.strip())
            label.setWordWrap(True)
            label_layout = QVBoxLayout()
            label_layout.addWidget(label)
            group.setLayout(label_layout)
            layout.addWidget(group)


def create_visualization_guide(parent=None):
    """Create a scrollable area containing all visualization explanations"""
    scroll = QScrollArea(parent)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    container = QWidget()
    layout = QVBoxLayout(container)

    viz_explanations = VisualizationExplanations()
    layout.addWidget(viz_explanations)

    scroll.setWidget(container)
    return scroll
