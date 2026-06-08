import sys
import serial
import serial.tools.list_ports
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QGridLayout
from PyQt5.QtGui import QIcon, QFont

class SerialReaderThread(QThread):
    data_received = pyqtSignal(str)

    def __init__(self, serial_connection):
        super().__init__()
        self.serial_connection = serial_connection
        self.running = True

    def run(self):
        while self.running:
            if self.serial_connection and self.serial_connection.is_open:
                try:
                    line = self.serial_connection.readline().decode().strip()
                    if line:
                        self.data_received.emit(line)
                except serial.SerialException as e:
                    self.data_received.emit(f"Error: {e}")
                except UnicodeDecodeError as e:
                    self.data_received.emit(f"Decode Error: {e}")

    def stop(self):
        self.running = False
        self.wait()

class CarControlApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Control")
        self.setGeometry(100, 100, 600, 400)

        # Initialize serial connection
        self.serial_connection = None
        self.serial_thread = None

        # Create UI elements
        self.create_ui()

    def create_ui(self):
        main_layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Disconnected", self)
        self.status_label.setFont(QFont("Arial", 14))
        main_layout.addWidget(self.status_label, alignment=Qt.AlignCenter)

        # Connect button
        connect_button = QPushButton("Connect", self)
        connect_button.setFont(QFont("Arial", 12))
        connect_button.setIcon(QIcon('icons/connect.png'))  # Add an icon
        connect_button.clicked.connect(self.connect_to_car)
        connect_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        main_layout.addWidget(connect_button, alignment=Qt.AlignCenter)

        # Remote control panel
        remote_control_frame = QWidget()
        remote_control_layout = QGridLayout(remote_control_frame)
        remote_control_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 10px; padding: 10px;")
        main_layout.addWidget(remote_control_frame)

        remote_control_layout.addWidget(QLabel("Remote Control", font=QFont("Arial", 16)), 0, 1, alignment=Qt.AlignCenter)

        button_style = """
            QPushButton {
                background-color: #A9A9A9;  /* Gray color */
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #696969;  /* Dark gray color */
            }
        """

        forward_button = QPushButton("Forward", self)
        forward_button.setFont(QFont("Arial", 12))
        forward_button.setIcon(QIcon('icons/forward.png'))
        forward_button.clicked.connect(lambda: self.send_command('@CAR_F\r\n'))
        forward_button.setStyleSheet(button_style)
        remote_control_layout.addWidget(forward_button, 1, 1)

        backward_button = QPushButton("Backward", self)
        backward_button.setFont(QFont("Arial", 12))
        backward_button.setIcon(QIcon('icons/backward.png'))
        backward_button.clicked.connect(lambda: self.send_command('@CAR_B\r\n'))
        backward_button.setStyleSheet(button_style)
        remote_control_layout.addWidget(backward_button, 3, 1)

        left_button = QPushButton("Turn Left", self)
        left_button.setFont(QFont("Arial", 12))
        left_button.setIcon(QIcon('icons/left.png'))
        left_button.clicked.connect(lambda: self.send_command('@CAR_TF\r\n'))
        left_button.setStyleSheet(button_style)
        remote_control_layout.addWidget(left_button, 2, 0)

        right_button = QPushButton("Turn Right", self)
        right_button.setFont(QFont("Arial", 12))
        right_button.setIcon(QIcon('icons/right.png'))
        right_button.clicked.connect(lambda: self.send_command('@CAR_TR\r\n'))
        right_button.setStyleSheet(button_style)
        remote_control_layout.addWidget(right_button, 2, 2)

        trans_left_button = QPushButton("Trans Left", self)
        trans_left_button.setFont(QFont("Arial", 12))
        trans_left_button.setIcon(QIcon('icons/trans_left.png'))
        trans_left_button.clicked.connect(lambda: self.send_command('@CAR_TSF\r\n'))
        trans_left_button.setStyleSheet(button_style)
        remote_control_layout.addWidget(trans_left_button, 1, 0)

        trans_right_button = QPushButton("Trans Right", self)
        trans_right_button.setFont(QFont("Arial", 12))
        trans_right_button.setIcon(QIcon('icons/trans_right.png'))
        trans_right_button.clicked.connect(lambda: self.send_command('@CAR_TSR\r\n'))
        trans_right_button.setStyleSheet(button_style)
        remote_control_layout.addWidget(trans_right_button, 1, 2)

        stop_button = QPushButton("Stop", self)
        stop_button.setFont(QFont("Arial", 12))
        stop_button.setIcon(QIcon('icons/stop.png'))
        stop_button.clicked.connect(lambda: self.send_command('@CAR_S\r\n'))
        stop_button.setStyleSheet(button_style)
        remote_control_layout.addWidget(stop_button, 2, 1)

        # Mode selection panel
        mode_selection_frame = QWidget()
        mode_selection_layout = QVBoxLayout(mode_selection_frame)
        mode_selection_frame.setStyleSheet("background-color: #e0e0e0; border-radius: 10px; padding: 10px;")
        main_layout.addWidget(mode_selection_frame)

        mode_selection_layout.addWidget(QLabel("Mode Selection", font=QFont("Arial", 16)), alignment=Qt.AlignCenter)

        remote_mode_button = QPushButton("Remote Mode", self)
        remote_mode_button.setFont(QFont("Arial", 12))
        remote_mode_button.clicked.connect(lambda: self.send_command('@CAR_Remote\r\n'))
        remote_mode_button.setStyleSheet(button_style)
        mode_selection_layout.addWidget(remote_mode_button, alignment=Qt.AlignCenter)

        follow_mode_button = QPushButton("Follow Mode", self)
        follow_mode_button.setFont(QFont("Arial", 12))
        follow_mode_button.clicked.connect(lambda: self.send_command('@CAR_Follow\r\n'))
        follow_mode_button.setStyleSheet(button_style)
        mode_selection_layout.addWidget(follow_mode_button, alignment=Qt.AlignCenter)

        line_mode_button = QPushButton("Line Mode", self)
        line_mode_button.setFont(QFont("Arial", 12))
        line_mode_button.clicked.connect(lambda: self.send_command('@CAR_Line\r\n'))
        line_mode_button.setStyleSheet(button_style)
        mode_selection_layout.addWidget(line_mode_button, alignment=Qt.AlignCenter)

        # Display panel
        display_frame = QWidget()
        display_layout = QVBoxLayout(display_frame)
        display_frame.setStyleSheet("background-color: #d0d0d0; border-radius: 10px; padding: 10px;")
        main_layout.addWidget(display_frame)

        display_layout.addWidget(QLabel("Display", font=QFont("Aria88l", 16)), alignment=Qt.AlignCenter)

        self.distance_label = QLabel("Distance: N/A", self)
        self.distance_label.setFont(QFont("Arial", 14))
        display_layout.addWidget(self.distance_label, alignment=Qt.AlignCenter)

        self.motor_status_label = QLabel("Motor Status: N/A", self)
        self.motor_status_label.setFont(QFont("Arial", 14))
        display_layout.addWidget(self.motor_status_label, alignment=Qt.AlignCenter)

        self.sensor_status_label = QLabel("Sensor Status: N/A", self)
        self.sensor_status_label.setFont(QFont("Arial", 14))
        display_layout.addWidget(self.sensor_status_label, alignment=Qt.AlignCenter)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def connect_to_car(self):
        try:
            # Specify the port directly for HC-06
            self.serial_connection = serial.Serial('COM8', 9600, timeout=1)
            self.status_label.setText(f"Connected to HC-06 on COM8")

            # Start serial reader thread
            self.serial_thread = SerialReaderThread(self.serial_connection)
            self.serial_thread.data_received.connect(self.update_display)
            self.serial_thread.start()
        except serial.SerialException as e:
            self.status_label.setText(f"Failed to connect: {e}")
            self.serial_connection = None

    def send_command(self, command):
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.write(command.encode())
                self.status_label.setText(f"Command '{command}' sent")
            except serial.SerialException as e:
                self.status_label.setText(f"Failed to send command: {e}")
        else:
            self.status_label.setText("Not connected")

    def update_display(self, message):
        # Directly display the message in the distance label
        if message.startswith("DIST:"):
            self.distance_label.setText(f"Distance: {message[5:]}")
        elif message.startswith("MOTOR:"):
            self.motor_status_label.setText(f"Motor Status: {message[6:]}")
        elif message.startswith("SENSOR:"):
            self.sensor_status_label.setText(f"Sensor Status: {message[7:]}")
        else:
            self.status_label.setText(message)

    def closeEvent(self, event):
        if self.serial_thread:
            self.serial_thread.stop()
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CarControlApp()
    window.show()
    sys.exit(app.exec_())