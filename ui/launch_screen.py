# Library imports
from PyQt6.QtWidgets import QWidget, QProgressBar, QLabel, QVBoxLayout, QPushButton
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QPainterPath, QLinearGradient
import math


class FibonacciSpiral(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.progress = 0
        self.set_progress(1)
        self.setMinimumSize(300, 300)

    def set_progress(self, value):
        self.progress = value
        self.update()

    def paintEvent(self, event):
        if self.progress == 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Center the drawing
        painter.translate(self.width() / 2, self.height() / 2)
        scale = min(self.width(), self.height()) / 500
        painter.scale(scale, scale)

        # Set up the pen
        pen = QPen(QColor(33, 150, 243))
        pen.setWidth(3)
        painter.setPen(pen)

        # Fibonacci numbers
        fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

        segments = int((len(fib) - 1) * (self.progress / 100))
        path = QPainterPath()
        x, y = 0, 0
        angle = 0
        scale_factor = 6

        for i in range(segments):
            radius = fib[i] * scale_factor
            center_x = x
            center_y = y

            if i == 0:
                path.moveTo(x, y)

            for t in range(91):
                rad = math.radians(t)
                arc_x = center_x + radius * math.cos(rad + angle)
                arc_y = center_y + radius * math.sin(rad + angle)

                if t == 0 and i == 0:
                    path.moveTo(arc_x, arc_y)
                else:
                    path.lineTo(arc_x, arc_y)

                if t == 90:
                    x = arc_x
                    y = arc_y

            angle += math.pi / 2

        # Draw glow effect
        for i in range(4):
            glow = QColor(33, 150, 243, 50 - i * 10)
            blur_pen = QPen(glow)
            blur_pen.setWidth(8 - i * 2)
            painter.setPen(blur_pen)
            painter.drawPath(path)

        # Draw main spiral
        painter.setPen(pen)
        painter.drawPath(path)


class LaunchScreen(QWidget):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.spiral = None
        self.progress = None
        self.status = None
        self.quote = None
        self.launch_btn = None
        self.progress_timer = None
        self.current_progress = 0
        self.quote_position = -800
        self.quote_shown = False
        self.quote_animation_timer = None

        # Set window style
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(800, 600)

        self.loading_steps = [
            "Initializing quantum systems...",
            "Loading harmonic analyzers...",
            "Preparing quantum circuits...",
            "Configuring visualization tools...",
            "Starting Quantum Harmonics...",
        ]

        # Create main layout first
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(30)
        self.main_layout.setContentsMargins(40, 150, 40, 40)
        self.setLayout(self.main_layout)

        # Initialize UI
        self.init_ui()
        self.setup_quote_animation()

    def init_ui(self):
        # Clear any existing widgets from layout
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add stretch at top
        self.main_layout.addStretch(1)

        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(30)

        # Title
        title = QLabel("Quantum Harmonics")
        title.setStyleSheet(
            """
            QLabel {
                color: white;
                font-size: 65px;
                font-weight: bold;
                font-family: Arial;
            }
        """
        )
        content_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        # Subtitle
        subtitle = QLabel("Advanced Quantum Music Analysis")
        subtitle.setStyleSheet(
            """
            QLabel {
                color: white;
                font-size: 30px;
                font-family: Arial;
                font-weight: bold;
            }
        """
        )
        content_layout.addWidget(subtitle, alignment=Qt.AlignmentFlag.AlignCenter)

        # Quote
        self.quote = QLabel(
            '"There is geometry in the humming of the strings,\n'
            'there is music in the spacing of the spheres."\n'
            "- Pythagoras"
        )
        self.quote.setStyleSheet(
            """
            QLabel {
                color: #B3E5FC;
                font-size: 18px;
                font-style: italic;
                font-family: Georgia, serif;
                margin-top: 20px;
            }
        """
        )
        self.quote.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quote.hide()
        content_layout.addWidget(self.quote)

        # Launch Button
        self.launch_btn = QPushButton("Launch Application")
        self.launch_btn.setFixedWidth(250)
        self.launch_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #3d405e;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 5px;
                font-size: 18px;
                font-family: Arial;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """
        )
        self.launch_btn.clicked.connect(self.start_loading)
        content_layout.addWidget(
            self.launch_btn, alignment=Qt.AlignmentFlag.AlignCenter
        )

        # Add the content widget to main layout
        self.main_layout.addWidget(
            content_widget, alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.main_layout.addStretch(1)

        # Progress bar (hidden initially)
        self.progress = QProgressBar()
        self.progress.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid #2196F3;
                border-radius: 5px;
                text-align: center;
                color: white;
                background-color: #1A237E;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                width: 15px;
                margin: 0.5px;
            }
        """
        )
        self.progress.hide()
        self.main_layout.addWidget(self.progress)

        # Status label (hidden initially)
        self.status = QLabel()
        self.status.setStyleSheet(
            """
            QLabel {
                color: white;
                font-family: Arial;
                font-size: 14px;
            }
        """
        )
        self.status.hide()
        self.main_layout.addWidget(self.status, alignment=Qt.AlignmentFlag.AlignCenter)

    def setup_quote_animation(self):
        self.quote_animation_timer = QTimer(self)
        self.quote_animation_timer.timeout.connect(self.animate_quote)
        self.quote_animation_timer.start(30)

    def animate_quote(self):
        if not self.quote_shown and not self.quote.isVisible():
            self.quote.show()
            self.quote_shown = True

        if self.quote_position < 0:
            self.quote_position += 20
            self.quote.setStyleSheet(
                f"""
                QLabel {{
                    color: #B3E5FC;
                    font-size: 18px;
                    font-style: italic;
                    margin-top: 20px;
                    margin-left: {self.quote_position}px;
                }}
            """
            )
        else:
            self.quote_animation_timer.stop()
            self.quote.setStyleSheet(
                """
                QLabel {
                    color: #B3E5FC;
                    font-size: 18px;
                    font-style: italic;
                    margin-top: 20px;
                    margin-left: 0px;
                }
            """
            )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Create gradient background
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(31, 47, 74))  # Dark blue
        gradient.setColorAt(1, QColor(21, 39, 69))  # Dark blue

        # Create rounded rectangle path
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 15, 15)

        # Fill background
        painter.fillPath(path, gradient)

    def start_loading(self):
        self.launch_btn.hide()
        self.quote.hide()

        # Store the progress bar and status label before clearing
        self.progress.setParent(None)  # Detach from current layout
        self.status.setParent(None)  # Detach from current layout

        # Clear current layout
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add Fibonacci spiral
        self.spiral = FibonacciSpiral()
        # self.spiral.setFixedSize(500, 500)
        self.main_layout.addWidget(self.spiral, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add spacer
        self.main_layout.addStretch(1)

        # Re-add and show progress bar and status
        self.progress.show()
        self.main_layout.addWidget(
            self.progress, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.status.show()
        self.main_layout.addWidget(self.status, alignment=Qt.AlignmentFlag.AlignCenter)

        # Start progress timer
        self.current_progress = 1
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(100)

    def update_progress(self):
        self.current_progress += 2
        if self.current_progress <= 100:
            self.progress.setValue(self.current_progress)
            self.spiral.set_progress(self.current_progress)
            step_index = min(len(self.loading_steps) - 1, self.current_progress // 20)
            self.status.setText(self.loading_steps[step_index])
        else:
            self.progress_timer.stop()
            self.cleanup()
            self.finished.emit()

    def cleanup(self):
        if self.progress_timer:
            self.progress_timer.stop()
        if self.quote_animation_timer:
            self.quote_animation_timer.stop()
