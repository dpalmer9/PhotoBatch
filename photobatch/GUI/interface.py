import sys
import pandas as pd
import configparser
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                               QPushButton, QLabel, QFileDialog, QLineEdit, QTableWidget,
                               QTableWidgetItem, QFormLayout, QLineEdit, QMessageBox,
                               QGroupBox, QCheckBox, QHBoxLayout, QMenuBar, QComboBox,
                               QListWidget, QListWidgetItem, QSpinBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction


class MultiSelectComboBox(QWidget):
    def __init__(self):
        super().__init__()

        # Layout to hold main elements
        layout = QVBoxLayout(self)

        # Line edit for direct entry
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Select or add new entries...")
        self.line_edit.returnPressed.connect(self.add_custom_entry)  # Custom entries

        # ListWidget for displaying items with checkboxes
        self.list_widget = QListWidget()

        # Add widgets to layout
        layout.addWidget(self.line_edit)
        layout.addWidget(self.list_widget)

    def add_option(self, option_text):
        """Adds an option to the list widget with a checkbox."""
        if option_text not in [self.list_widget.item(i).text() for i in range(self.list_widget.count())]:
            item = QListWidgetItem(option_text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # Add checkbox
            item.setCheckState(Qt.Unchecked)  # Default to unchecked
            self.list_widget.addItem(item)

    def add_custom_entry(self):
        """Adds a custom entry from the line edit to the list widget."""
        custom_text = self.line_edit.text().strip()
        if custom_text:
            self.add_option(custom_text)
            self.line_edit.clear()  # Clear text after entry

    def get_checked_items(self):
        """Returns a list of all checked items."""
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())
                if self.list_widget.item(i).checkState() == Qt.Checked]

class FiberPhotometryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fiber Photometry Data Analyzer")
        self.resize(1200, 800)  # Set a larger default window size

        # Initialize configuration
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        # Initialize template data
        self.template_loaded = False
        self.unique_event_types = []
        self.unique_event_names = []
        self.unique_event_groups = []
        self.trial_stage_options = []

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

        # Initialize menu bar
        self.init_menu_bar()

    def init_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # File menu
        file_menu = menu_bar.addMenu("File")

        # Import template behavior file action
        import_template_action = QAction("Import Template Behaviour File", self)
        import_template_action.triggered.connect(self.import_template_behaviour_file)
        file_menu.addAction(import_template_action)

    def import_template_behaviour_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Template Behaviour File", "", "CSV Files (*.csv)")
        column_index = 0
        if file_path:
            try:
                data = pd.read_csv(file_path, sep=',', header=None, names=range(17))
                for index, row in data.iterrows():
                    if row[0] == 'Evnt_Time':
                        column_index = index
                self.event_data = data.iloc[column_index:]
                self.event_data.columns = self.event_data.iloc[0]
                self.event_data = self.event_data.drop(column_index)

                # Extract unique values for dropdowns
                self.unique_event_types = self.event_data['Evnt_Name'].dropna().unique().tolist()
                self.unique_event_names = self.event_data['Item_Name'].dropna().unique().tolist()
                self.unique_event_groups = self.event_data['Group_ID'].dropna().unique().tolist()
                self.trial_stage_options = self.event_data.loc[self.event_data['Evnt_Name'] == "Condition Event", 'Item_Name'].dropna().unique().tolist()
                self.template_loaded = True

                # Update the trial start stage entry in options tab if template is loaded
                self.update_options_with_template()

                QMessageBox.information(self, "Imported", f"Template behaviour file imported from: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load template: {str(e)}")

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

        # Set column count and headers
        self.event_table.setColumnCount(5)
        self.event_table.setHorizontalHeaderLabels([
            'event_type', 'event_name', 'event_group', 'event_arg', 'num_filter'])

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
        self.file_pair_table.setColumnCount(6)
        self.file_pair_table.setHorizontalHeaderLabels([
            'abet_path', 'doric_path', 'ctrl_col_num', 'act_col_num', 'ttl_col_num', 'mode'
        ])
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
        data = pd.read_csv(file_path)
        rows, cols = data.shape
        table_widget.setRowCount(rows)
        table_widget.setColumnCount(cols)
        table_widget.setHorizontalHeaderLabels(data.columns)

        for row in range(rows):
            for col in range(cols):
                header = data.columns[col].lower()
                if header == 'event_type' and self.template_loaded:
                    event_type_combo = QComboBox()
                    event_type_combo.addItems(self.unique_event_types)
                    event_type_combo.setCurrentText(str(data.iat[row, col]))
                    event_type_combo.currentTextChanged.connect(
                        lambda text, r=row: self.filter_event_data(r, text, table_widget))
                    table_widget.setCellWidget(row, col, event_type_combo)
                elif header == 'event_name' and self.template_loaded:
                    event_name_combo = QComboBox()
                    event_name_combo.currentTextChanged.connect(
                        lambda text, r=row: self.filter_event_group(r, text, table_widget))
                    table_widget.setCellWidget(row, col, event_name_combo)
                elif header == 'event_group' and self.template_loaded:
                    event_group_combo = QComboBox()
                    table_widget.setCellWidget(row, col, event_group_combo)
                elif header == 'num_filter':
                    num_filter_spinbox = QSpinBox()
                    num_filter_spinbox.setValue(int(data.iat[row, col]) if pd.notna(data.iat[row, col]) else 0)
                    num_filter_spinbox.valueChanged.connect(
                        lambda value, r=row: self.adjust_filter_columns(r, value, table_widget))
                    table_widget.setCellWidget(row, col, num_filter_spinbox)
                else:
                    item = QTableWidgetItem(str(data.iat[row, col]))
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    table_widget.setItem(row, col, item)

        table_widget.resizeColumnsToContents()

    def filter_event_data(self, row, event_type, table_widget):
        filtered_names = self.event_data[self.event_data['Evnt_Name'] == event_type]['Item_Name'].dropna().unique()
        name_combo = table_widget.cellWidget(row, table_widget.columnCount() - 22)  # Column index for event_name
        name_combo.clear()
        name_combo.addItems(filtered_names)

        # Trigger the event group filter for the first available event name
        if filtered_names.size > 0:
            self.filter_event_group(row, filtered_names[0], table_widget)

    def filter_event_group(self, row, event_name, table_widget):
        event_type_combo = table_widget.cellWidget(row, table_widget.columnCount() - 23)  # Column index for event_type
        event_type = event_type_combo.currentText()

        filtered_groups = self.event_data[
            (self.event_data['Evnt_Name'] == event_type) &
            (self.event_data['Item_Name'] == event_name)
            ]['Group_ID'].dropna().unique()

        group_combo = table_widget.cellWidget(row, table_widget.columnCount() - 21)  # Column index for event_group
        group_combo.clear()
        group_combo.addItems(filtered_groups)

    def add_row(self, table_widget):
        current_row_count = table_widget.rowCount()
        table_widget.insertRow(current_row_count)

        for col in range(table_widget.columnCount()):
            header = table_widget.horizontalHeaderItem(col).text().lower()
            if header == 'event_type' and self.template_loaded:
                event_type_combo = QComboBox()
                event_type_combo.addItems(self.unique_event_types)
                event_type_combo.currentTextChanged.connect(
                    lambda text, r=current_row_count: self.filter_event_data(r, text, table_widget))
                table_widget.setCellWidget(current_row_count, col, event_type_combo)
            elif header == 'event_name' and self.template_loaded:
                event_name_combo = QComboBox()
                event_name_combo.currentTextChanged.connect(
                    lambda text, r=current_row_count: self.filter_event_group(r, text, table_widget))
                table_widget.setCellWidget(current_row_count, col, event_name_combo)
            elif header == 'event_group' and self.template_loaded:
                event_group_combo = QComboBox()
                table_widget.setCellWidget(current_row_count, col, event_group_combo)
            elif header == 'num_filter':
                num_filter_spinbox = QSpinBox()
                num_filter_spinbox.setValue(0)
                num_filter_spinbox.valueChanged.connect(
                    lambda value, r=current_row_count: self.adjust_filter_columns(r, value, table_widget))
                table_widget.setCellWidget(current_row_count, col, num_filter_spinbox)
            else:
                item = QTableWidgetItem("")
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                table_widget.setItem(current_row_count, col, item)

    def adjust_filter_columns(self, row, num_filters, table_widget):
        """Adjust columns in the table based on the num_filter value."""
        # Clear any existing filter columns beyond the current num_filter count
        total_columns = 5 + (num_filters * 6)
        table_widget.setColumnCount(total_columns)

        # Set headers for the filter columns
        for i in range(1, num_filters + 1):
            base_index = 5 + (i - 1) * 6
            if i == 1:
                headers = ['filter_type', 'filter_name', 'filter_group', 'filter_arg', 'filter_eval', 'filter_prior']
            else:
                headers = [f'filter_type{i}', f'filter_name{i}', f'filter_group{i}', f'filter_arg{i}',
                           f'filter_eval{i}', f'filter_prior{i}']
            for j, header in enumerate(headers):
                table_widget.setHorizontalHeaderItem(base_index + j, QTableWidgetItem(header))
                item = QTableWidgetItem("")
                table_widget.setItem(row, base_index + j, item)  # Initialize with empty items

        # Adjust the table size
        table_widget.resizeColumnsToContents()

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
                row_data = []
                for col in range(cols):
                    if isinstance(table_widget.cellWidget(row, col), QComboBox):
                        row_data.append(table_widget.cellWidget(row, col).currentText())
                    else:
                        item = table_widget.item(row, col)
                        row_data.append(item.text() if item else "")
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
        multibox_options = ['trial_start_stage','trial_end_stage','exclusion_list']

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
                    elif key in multibox_options:
                        multicombobox_edit = MultiSelectComboBox()
                        setattr(self, f"{section}_{key}_multicombobox_edit", multicombobox_edit)  # Store line edit for saving
                        group_layout.addRow(display_key, getattr(self, f"{section}_{key}_multicombobox_edit"))
                    else:
                        # Use line edit for other parameters
                        line_edit = QLineEdit(value)
                        setattr(self, f"{section}_{key}_line_edit", line_edit)  # Store line edit for saving
                        group_layout.addRow(display_key, getattr(self, f"{section}_{key}_line_edit"))

                group_box.setLayout(group_layout)
                main_layout.addWidget(group_box)

        # Save Button
        save_button = QPushButton("Save Changes")
        save_button.clicked.connect(self.save_config_changes)
        main_layout.addWidget(save_button)

        self.options_tab.setLayout(main_layout)

    def update_options_with_template(self):
        if self.template_loaded:
            for stage in self.trial_stage_options:
                try:
                    getattr(self, 'ITI_Window_trial_start_stage_multicombobox_edit').add_option(stage)
                    getattr(self, 'ITI_Window_trial_end_stage_multicombobox_edit').add_option(stage)
                    getattr(self, 'Filter_exclusion_list_multicombobox_edit').add_option(stage)
                except:
                    print('No files loaded')
        else:
            QMessageBox.warning(self, "Error", "No template loaded, cannot update trial start stage.")

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
        with open('../Config.ini', 'w') as configfile:
            self.config.write(configfile)

        # Confirmation message
        QMessageBox.information(self, "Saved", "Configuration changes saved successfully.")


#if __name__ == "__main__":
    #app = QApplication(sys.argv)
    #window = FiberPhotometryApp()
    #window.show()
    #sys.exit(app.exec())
