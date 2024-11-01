import sys
import pandas as pd
import configparser
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                               QPushButton, QLabel, QFileDialog, QLineEdit, QTableWidget,
                               QTableWidgetItem, QFormLayout, QLineEdit, QMessageBox,
                               QGroupBox, QCheckBox, QHBoxLayout)
from PySide6.QtCore import Qt


class FiberPhotometryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fiber Photometry Data Analyzer")
        self.resize(1200, 800)  # Set a larger default window size

        # Initialize configuration
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        # Main layout with tabs
        self.tabs = QTabWidget()

        self.home_tab = QWidget()
        self.event_sheet_tab = QWidget()
        self.file_pair_tab = QWidget()
        self.options_tab = QWidget()
        self.analysis_tab = QWidget()
        self.results_tab = QWidget()

        self.tabs.addTab(self.home_tab, "Home")
        self.tabs.addTab(self.event_sheet_tab, "Event Sheet")
        self.tabs.addTab(self.file_pair_tab, "File Pair")
        self.tabs.addTab(self.options_tab, "Options")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.results_tab, "Results")

        self.setCentralWidget(self.tabs)

        # Initialize tab layouts
        self.init_home_tab()
        self.init_event_sheet_tab()
        self.init_file_pair_tab()
        self.init_analysis_tab()
        self.init_results_tab()
        self.init_options_tab()

    def init_home_tab(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Welcome to the Fiber Photometry Data Analyzer"))
        self.home_tab.setLayout(layout)

    def init_event_sheet_tab(self):
        layout = QVBoxLayout()

        # File selection and table for Event Sheet
        self.event_file_btn = QPushButton("Select Event Sheet")
        self.event_file_btn.clicked.connect(self.load_event_sheet)
        self.event_file_path = QLineEdit()
        layout.addWidget(QLabel("Event Sheet File:"))
        layout.addWidget(self.event_file_btn)
        layout.addWidget(self.event_file_path)

        # Table to display the Event Sheet content
        self.event_table = QTableWidget()
        layout.addWidget(self.event_table)

        # Buttons to add/remove rows and save
        button_layout = QHBoxLayout()
        add_row_btn = QPushButton("Add Row")
        add_row_btn.clicked.connect(lambda: self.add_row(self.event_table))
        remove_row_btn = QPushButton("Remove Row")
        remove_row_btn.clicked.connect(lambda: self.remove_row(self.event_table))
        save_btn = QPushButton("Save to CSV")
        save_btn.clicked.connect(lambda: self.save_table_to_csv(self.event_table))
        button_layout.addWidget(add_row_btn)
        button_layout.addWidget(remove_row_btn)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)

        self.event_sheet_tab.setLayout(layout)

    def init_file_pair_tab(self):
        layout = QVBoxLayout()

        # File selection and table for File Pair
        self.file_pair_btn = QPushButton("Select File Pair Sheet")
        self.file_pair_btn.clicked.connect(self.load_file_pair_sheet)
        self.file_pair_path = QLineEdit()
        layout.addWidget(QLabel("File Pair Sheet File:"))
        layout.addWidget(self.file_pair_btn)
        layout.addWidget(self.file_pair_path)

        # Table to display the File Pair content
        self.file_pair_table = QTableWidget()
        layout.addWidget(self.file_pair_table)

        # Buttons to add/remove rows and save
        button_layout = QHBoxLayout()
        add_row_btn = QPushButton("Add Row")
        add_row_btn.clicked.connect(lambda: self.add_row(self.file_pair_table))
        remove_row_btn = QPushButton("Remove Row")
        remove_row_btn.clicked.connect(lambda: self.remove_row(self.file_pair_table))
        save_btn = QPushButton("Save to CSV")
        save_btn.clicked.connect(lambda: self.save_table_to_csv(self.file_pair_table))
        button_layout.addWidget(add_row_btn)
        button_layout.addWidget(remove_row_btn)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)

        self.file_pair_tab.setLayout(layout)

    def load_event_sheet(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Event Sheet", "", "CSV Files (*.csv)")
        if file_path:
            self.event_file_path.setText(file_path)
            self.display_csv_in_table(file_path, self.event_table)

    def load_file_pair_sheet(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File Pair Sheet", "", "CSV Files (*.csv)")
        if file_path:
            self.file_pair_path.setText(file_path)
            self.display_csv_in_table(file_path, self.file_pair_table)

    def display_csv_in_table(self, file_path, table_widget):
        # Load CSV data
        data = pd.read_csv(file_path)
        rows, cols = data.shape

        # Set up the table widget
        table_widget.setRowCount(rows)
        table_widget.setColumnCount(cols)
        table_widget.setHorizontalHeaderLabels(data.columns)

        # Populate the table with CSV data
        for row in range(rows):
            for col in range(cols):
                item = QTableWidgetItem(str(data.iat[row, col]))
                item.setFlags(item.flags() | Qt.ItemIsEditable)  # Make cells editable
                table_widget.setItem(row, col, item)

        table_widget.resizeColumnsToContents()

    def add_row(self, table_widget):
        current_row_count = table_widget.rowCount()
        table_widget.insertRow(current_row_count)
        for col in range(table_widget.columnCount()):
            item = QTableWidgetItem("")
            item.setFlags(item.flags() | Qt.ItemIsEditable)  # Make new cells editable
            table_widget.setItem(current_row_count, col, item)

    def remove_row(self, table_widget):
        current_row = table_widget.currentRow()
        if current_row != -1:
            table_widget.removeRow(current_row)

    def save_table_to_csv(self, table_widget):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Table to CSV", "", "CSV Files (*.csv)")
        if file_path:
            # Extract data from the table
            rows = table_widget.rowCount()
            cols = table_widget.columnCount()
            data = []
            headers = [table_widget.horizontalHeaderItem(col).text() for col in range(cols)]
            for row in range(rows):
                row_data = [table_widget.item(row, col).text() if table_widget.item(row, col) else "" for col in
                            range(cols)]
                data.append(row_data)

            # Save data to CSV using pandas
            df = pd.DataFrame(data, columns=headers)
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Saved", "Table saved successfully to CSV.")

    def init_analysis_tab(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Configure and Run Analysis"))
        self.analysis_tab.setLayout(layout)

    def init_results_tab(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Analysis Results"))
        self.results_tab.setLayout(layout)

    def init_options_tab(self):
        main_layout = QVBoxLayout()

        # Filter out file and event sheet sections
        excluded_sections = {'Filepath', 'Event_Window'}

        # Create a group for each section and add its fields
        for section in self.config.sections():
            if section not in excluded_sections:
                group_box = QGroupBox(section.replace("_", " "))  # Section title
                group_layout = QFormLayout()

                # Create fields for each parameter within the section
                for key, value in self.config[section].items():
                    display_key = key.replace("_", " ")  # Replace underscores with spaces

                    if section == "Output":
                        # Use a checkbox for Output parameters
                        checkbox = QCheckBox()
                        checkbox.setChecked(value.lower() == 'true')
                        group_layout.addRow(display_key, checkbox)
                        setattr(self, f"{section}_{key}_checkbox", checkbox)  # Store checkbox for saving
                    else:
                        # Use line edit for other parameters
                        line_edit = QLineEdit(value)
                        group_layout.addRow(display_key, line_edit)
                        setattr(self, f"{section}_{key}_line_edit", line_edit)  # Store line edit for saving

                group_box.setLayout(group_layout)
                main_layout.addWidget(group_box)

        # Save Button
        save_button = QPushButton("Save Changes")
        save_button.clicked.connect(self.save_config_changes)
        main_layout.addWidget(save_button)

        self.options_tab.setLayout(main_layout)

    def save_config_changes(self):
        # Update config with new values from form
        for section in self.config.sections():
            for key in self.config[section]:
                widget = getattr(self, f"{section}_{key}_line_edit", None) or getattr(self, f"{section}_{key}_checkbox",
                                                                                      None)
                if isinstance(widget, QLineEdit):
                    self.config[section][key] = widget.text()
                elif isinstance(widget, QCheckBox):
                    self.config[section][key] = 'true' if widget.isChecked() else 'false'

        # Save to config.ini file
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

        # Confirmation message
        QMessageBox.information(self, "Saved", "Configuration changes saved successfully.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FiberPhotometryApp()
    window.show()
    sys.exit(app.exec())