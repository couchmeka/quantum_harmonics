from PyQt6.QtGui import QPainter, QLinearGradient, QPainterPath, QColor
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QGridLayout,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QRectF
import qtawesome as qta
from ui.features.ai_ui.ai_analysis_screen import AIAnalysisScreen
from ui.features.quantum_ui.harmonics_screen import HarmonicsScreen


class FeatureScreen(QWidget):
    def __init__(self, quantum_tabs=None):
        super().__init__()
        if quantum_tabs is None:
            raise ValueError("quantum_tabs cannot be None")
        self.quantum_tabs = quantum_tabs
        self.setMinimumSize(800, 600)  # Minimum size
        self.setWindowTitle("Quantum Harmonics")

        # Allow window to be resized
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.setup_ui()

    def setup_ui(self):
        # Title
        title = QLabel("Quantum Harmonics")
        title.setStyleSheet(
            """
            QLabel {
                color: white;
                font-size: 48px;
                font-weight: bold;
                font-family: Arial;
                margin: 20px;
            }
        """
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(title)

        # Create grid for buttons
        grid_layout = QGridLayout()
        grid_layout.setSpacing(20)

        # Define all features with their icons
        # Row 1 starts with 0
        # Row 2 starts with 1
        features = [
            ("Harmonics", "fa5s.music", self.open_harmonics, 0, 0),
            ("Light", "fa5s.lightbulb", self.open_light, 0, 1),
            ("AI", "fa5s.brain", self.open_ai, 1, 0),
            ("Cryptography", "fa5s.puzzle-piece", self.open_cryptography, 1, 1),
        ]

        for name, icon_name, callback, row, col in features:
            button = self.create_feature_button(name, icon_name, callback)
            grid_layout.addWidget(button, row, col)

        # Add grid to main layout
        self.main_layout.addLayout(grid_layout)
        self.main_layout.addStretch()

    def create_feature_button(self, name, icon_name, callback):
        button = QPushButton()
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        button.setFixedSize(250, 150)  # Fixed size for consistent grid

        # Create icon
        icon = qta.icon(icon_name, color="white")

        # Create vertical layout for button content(most recent change)
        button_layout = QGridLayout(button)
        button_layout.setSpacing(15)
        button_layout.setContentsMargins(20, 20, 20, 20)

        # Create horizontal layout for button creation

        # Add icon
        icon_label = QLabel()
        icon_label.setPixmap(icon.pixmap(64, 64))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(icon_label)

        # Add text
        text_label = QLabel(name)
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        button_layout.addWidget(text_label)

        # Style the button
        button.setStyleSheet(
            """
            QPushButton {
                background-color: #3d405e;
                border: 2px solid #4a4d78;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #1976D2;
                border: 2px solid #2196F3;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """
        )

        button.clicked.connect(callback)
        return button

    def open_harmonics(self):
        try:
            harmonics_screen = HarmonicsScreen(
                quantum_tabs=self.quantum_tabs,
                data_manager=getattr(self, "data_manager", None),
            )

            if harmonics_screen is None:
                raise ValueError("Failed to create HarmonicsScreen")

            # Show the new window
            harmonics_screen.show()

            # Close the feature screen
            self.close()

        except Exception as e:
            print(f"Error setting up Harmonics: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to open Harmonics: {str(e)}")

    def open_ai(self):
        try:
            self.raise_()
            self.activateWindow()

            while self.main_layout.count():
                child = self.main_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            # Use the data_manager passed from MainApplication
            ai_screen = AIAnalysisScreen(data_manager=self.data_manager)
            self.main_layout.addWidget(ai_screen)

            self.setWindowFlags(
                Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint
            )
            self.show()

        except Exception as e:
            print(f"Error setting up AI Analysis: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to open AI Analysis: {str(e)}")

    def open_light(self):
        QMessageBox.information(
            self, "Coming Soon", "Light module is under development"
        )

    def open_cryptography(self):
        QMessageBox.information(
            self, "Coming Soon", "Cryptography's algorithm module is under development"
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(31, 47, 74))
        gradient.setColorAt(1, QColor(21, 39, 69))

        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 15, 15)
        painter.fillPath(path, gradient)
