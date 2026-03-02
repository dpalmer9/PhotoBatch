import sys
import pandas as pd
import os
import configparser
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                               QPushButton, QLabel, QFileDialog, QLineEdit, QTableWidget,
                               QTableWidgetItem, QFormLayout, QMessageBox,
                               QGroupBox, QCheckBox, QHBoxLayout, QMenuBar, QComboBox,
                               QListWidget, QListWidgetItem, QSpinBox, QScrollArea)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction
from functools import partial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from Processing import data_processor

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)

class MultiSelectComboBox(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Select or add new entries...")
        self.line_edit.returnPressed.connect(self.add_custom_entry)
        self.list_widget = QListWidget()
        layout.addWidget(self.line_edit)
        layout.addWidget(self.list_widget)

    def add_option(self, option_text):
        if option_text not in [self.list_widget.item(i).text() for i in range(self.list_widget.count())]:
            item = QListWidgetItem(option_text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

    def add_custom_entry(self):
        custom_text = self.line_edit.text().strip()
        if custom_text:
            self.add_option(custom_text)
            self.line_edit.clear()

    def get_checked_items(self):
        return [self.list_widget.item(i).text() for i in range(self.list_widget.count())
                if self.list_widget.item(i).checkState() == Qt.Checked]

class FilePathWidget(QWidget):
    """A composite widget with a QLineEdit and a browse button ('...') for file path input."""
    def __init__(self, initial_text="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.line_edit = QLineEdit(initial_text)
        self.browse_btn = QPushButton("\u2026")
        self.browse_btn.setFixedWidth(28)
        self.browse_btn.setToolTip("Browse for file")
        self.browse_btn.clicked.connect(self._browse)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.browse_btn)

    def _browse(self):
        start_dir = os.path.dirname(self.line_edit.text()) if self.line_edit.text() else ""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", start_dir, "All Files (*)")
        if file_path:
            self.line_edit.setText(file_path)

    def text(self):
        return self.line_edit.text()

    def setText(self, text):
        self.line_edit.setText(text)


class AnalysisThread(QThread):
    """Background thread that runs process_files so the GUI stays responsive
    during potentially long multi-file analyses.
    """
    results_ready = Signal(list, dict)   # (results, combined_results)
    error_occurred = Signal(str)

    def __init__(self, file_pair_path, event_sheet_path, output_selections,
                 config, num_workers, parent=None):
        super().__init__(parent)
        self.file_pair_path = file_pair_path
        self.event_sheet_path = event_sheet_path
        self.output_selections = output_selections
        self.config = config
        self.num_workers = num_workers

    def run(self):
        try:
            results, combined_results = data_processor.process_files(
                self.file_pair_path,
                self.event_sheet_path,
                self.output_selections,
                self.config,
                self.num_workers
            )
            self.results_ready.emit(results, combined_results)
        except Exception as e:
            self.error_occurred.emit(str(e))


class FiberPhotometryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fiber Photometry Data Analyzer")
        self.resize(1200, 800)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file_path = os.path.normpath(os.path.join(script_dir, '..', 'Config.ini'))
        self.config = configparser.ConfigParser()

        self.analysis_results = []
        self.combined_results = {}
        self.plot_data = pd.DataFrame()

        try:
            if not self.config.read(self.config_file_path):
                QMessageBox.warning(self, "Config File Warning", f"Config file '{self.config_file_path}' not found or empty. Defaults may not be loaded correctly.")
        except configparser.Error as e:
            QMessageBox.critical(self, "Config File Error", f"Error reading config file '{self.config_file_path}': {e}")

        # Capture the original configuration text so we can detect changes and prompt to save on exit
        try:
            import io
            if os.path.exists(self.config_file_path):
                try:
                    with open(self.config_file_path, 'r', encoding='utf-8') as f:
                        self._original_config_text = f.read()
                except Exception:
                    # Fallback to writing current parser state
                    buf = io.StringIO()
                    self.config.write(buf)
                    self._original_config_text = buf.getvalue()
            else:
                buf = io.StringIO()
                self.config.write(buf)
                self._original_config_text = buf.getvalue()
        except Exception:
            self._original_config_text = None

        self.template_loaded = False
        self.tabs = QTabWidget()
        self.home_tab = QWidget()
        self.event_sheet_tab = QWidget()
        self.file_pair_tab = QWidget()
        self.options_tab = QWidget()
        self.analysis_tab = QWidget()
        self.results_tab = QWidget()
        self.visualization_tab = QWidget()

        self.tabs.addTab(self.home_tab, "Home")
        self.tabs.addTab(self.event_sheet_tab, "Event Sheet")
        self.tabs.addTab(self.file_pair_tab, "File Pair")
        self.tabs.addTab(self.options_tab, "Options")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.visualization_tab, "Visualization")

        self.setCentralWidget(self.tabs)
        self.init_home_tab()
        self.init_event_sheet_tab()
        self.init_file_pair_tab()
        self.init_analysis_tab()
        self.init_results_tab()
        self.init_options_tab()
        self.init_visualization_tab()
        self.init_menu_bar()
        # After all UI tabs/widgets are initialized, load any default file paths from the config
        # and auto-populate the event and file-pair tables if valid file paths are provided.
        self.load_defaults_from_config()

    def init_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        file_menu = menu_bar.addMenu("File")
        import_template_action = QAction("Import Template Behaviour File", self)
        import_template_action.triggered.connect(self.import_template_behaviour_file)
        file_menu.addAction(import_template_action)
        load_config_action = QAction("Load Configuration...", self)
        load_config_action.triggered.connect(self.load_configuration_file)
        file_menu.addAction(load_config_action)
        save_config_as_action = QAction("Save Configuration As...", self)
        save_config_as_action.triggered.connect(self.save_configuration_as)
        file_menu.addAction(save_config_as_action)

    def import_template_behaviour_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Template Behaviour File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                data = pd.read_csv(file_path, sep=',', header=None, names=range(17))
                column_index = 0
                for index, row in data.iterrows():
                    if row[0] == 'Evnt_Time':
                        column_index = index
                self.event_data = data.iloc[column_index:]
                self.event_data.columns = self.event_data.iloc[0]
                self.event_data = self.event_data.drop(column_index)
                self.unique_event_types = self.event_data['Evnt_Name'].dropna().unique().tolist()
                self.unique_event_names = self.event_data['Item_Name'].dropna().unique().tolist()
                self.unique_event_groups = self.event_data['Group_ID'].dropna().unique().tolist()
                self.trial_stage_options = self.event_data.loc[self.event_data['Evnt_Name'] == "Condition Event", 'Item_Name'].dropna().unique().tolist()
                self.template_loaded = True
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
        self.event_file_btn = QPushButton("Select Event Sheet")
        self.event_file_btn.clicked.connect(self.load_event_sheet)
        self.event_file_path = QLineEdit()
        layout.addWidget(QLabel("Event Sheet File:"))
        layout.addWidget(self.event_file_btn)
        layout.addWidget(self.event_file_path)
        self.event_table = QTableWidget()
        self.event_table.setColumnCount(5)
        self.event_table.setHorizontalHeaderLabels(['event_type', 'event_name', 'event_group', 'event_arg', 'num_filter'])
        layout.addWidget(self.event_table)
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
        self.file_pair_btn = QPushButton("Select File Pair Sheet")
        self.file_pair_btn.clicked.connect(self.load_file_pair_sheet)
        self.file_pair_path = QLineEdit()
        layout.addWidget(QLabel("File Pair Sheet File:"))
        layout.addWidget(self.file_pair_btn)
        layout.addWidget(self.file_pair_path)
        self.file_pair_table = QTableWidget()
        self.file_pair_table.setColumnCount(6)
        self.file_pair_table.setHorizontalHeaderLabels(['abet_path', 'doric_path', 'ctrl_col_num', 'act_col_num', 'ttl_col_num', 'mode'])
        layout.addWidget(self.file_pair_table)
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
                    event_type_combo.currentTextChanged.connect(lambda text, r=row: self.filter_event_data(r, text, table_widget))
                    table_widget.setCellWidget(row, col, event_type_combo)
                elif header == 'event_name' and self.template_loaded:
                    event_name_combo = QComboBox()
                    event_name_combo.currentTextChanged.connect(lambda text, r=row: self.filter_event_group(r, text, table_widget))
                    table_widget.setCellWidget(row, col, event_name_combo)
                elif header == 'event_group' and self.template_loaded:
                    event_group_combo = QComboBox()
                    table_widget.setCellWidget(row, col, event_group_combo)
                elif header == 'num_filter':
                    num_filter_spinbox = QSpinBox()
                    num_filter_spinbox.setValue(int(data.iat[row, col]) if pd.notna(data.iat[row, col]) else 0)
                    num_filter_spinbox.valueChanged.connect(lambda value, r=row: self.adjust_filter_columns(r, value, table_widget))
                    table_widget.setCellWidget(row, col, num_filter_spinbox)
                elif header in ('abet_path', 'doric_path'):
                    cell_text = str(data.iat[row, col]) if pd.notna(data.iat[row, col]) else ""
                    fp_widget = FilePathWidget(cell_text)
                    table_widget.setCellWidget(row, col, fp_widget)
                else:
                    item = QTableWidgetItem(str(data.iat[row, col]))
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    table_widget.setItem(row, col, item)
        # Ensure columns and rows are sized to show content and refresh the widget so rows become visible
        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()
        try:
            table_widget.scrollToTop()
        except Exception:
            pass
        table_widget.repaint()
        table_widget.setVisible(True)

    def filter_event_data(self, row, event_type, table_widget):
        filtered_names = self.event_data[self.event_data['Evnt_Name'] == event_type]['Item_Name'].dropna().unique()
        name_combo = table_widget.cellWidget(row, 1)
        name_combo.clear()
        name_combo.addItems(filtered_names)
        if filtered_names.size > 0:
            self.filter_event_group(row, filtered_names[0], table_widget)

    def filter_event_group(self, row, event_name, table_widget):
        event_type_combo = table_widget.cellWidget(row, 0)
        event_type = event_type_combo.currentText()
        filtered_groups = self.event_data[(self.event_data['Evnt_Name'] == event_type) & (self.event_data['Item_Name'] == event_name)]['Group_ID'].dropna().unique()
        group_combo = table_widget.cellWidget(row, 2)
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
                event_type_combo.currentTextChanged.connect(lambda text, r=current_row_count: self.filter_event_data(r, text, table_widget))
                table_widget.setCellWidget(current_row_count, col, event_type_combo)
            elif header == 'event_name' and self.template_loaded:
                event_name_combo = QComboBox()
                event_name_combo.currentTextChanged.connect(lambda text, r=current_row_count: self.filter_event_group(r, text, table_widget))
                table_widget.setCellWidget(current_row_count, col, event_name_combo)
            elif header == 'event_group' and self.template_loaded:
                event_group_combo = QComboBox()
                table_widget.setCellWidget(current_row_count, col, event_group_combo)
            elif header == 'num_filter':
                num_filter_spinbox = QSpinBox()
                num_filter_spinbox.setValue(0)
                num_filter_spinbox.valueChanged.connect(lambda value, r=current_row_count: self.adjust_filter_columns(r, value, table_widget))
                table_widget.setCellWidget(current_row_count, col, num_filter_spinbox)
            elif header in ('abet_path', 'doric_path'):
                fp_widget = FilePathWidget()
                table_widget.setCellWidget(current_row_count, col, fp_widget)
            else:
                item = QTableWidgetItem("")
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                table_widget.setItem(current_row_count, col, item)

    def adjust_filter_columns(self, row, num_filters, table_widget):
        total_columns = 5 + (num_filters * 6)
        table_widget.setColumnCount(total_columns)
        for i in range(1, num_filters + 1):
            base_index = 5 + (i - 1) * 6
            headers = [f'filter_type{i}', f'filter_name{i}', f'filter_group{i}', f'filter_arg{i}', f'filter_eval{i}', f'filter_prior{i}']
            for j, header in enumerate(headers):
                table_widget.setHorizontalHeaderItem(base_index + j, QTableWidgetItem(header))
                item = QTableWidgetItem("")
                table_widget.setItem(row, base_index + j, item)
        table_widget.resizeColumnsToContents()

    def remove_row(self, table_widget):
        current_row = table_widget.currentRow()
        if current_row != -1:
            table_widget.removeRow(current_row)

    def save_table_to_csv(self, table_widget):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Table to CSV", "", "CSV Files (*.csv)")
        if file_path:
            rows = table_widget.rowCount()
            cols = table_widget.columnCount()
            data = []
            headers = [table_widget.horizontalHeaderItem(col).text() for col in range(cols)]
            for row in range(rows):
                row_data = []
                for col in range(cols):
                    widget = table_widget.cellWidget(row, col)
                    if isinstance(widget, QComboBox):
                        row_data.append(widget.currentText())
                    elif isinstance(widget, FilePathWidget):
                        row_data.append(widget.text())
                    else:
                        item = table_widget.item(row, col)
                        row_data.append(item.text() if item else "")
                data.append(row_data)
            df = pd.DataFrame(data, columns=headers)
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Saved", "Table saved successfully to CSV.")

    def init_analysis_tab(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Configure Analysis Options and Run"))
        if "Output" in self.config:
            output_group_box = QGroupBox("Output")
            output_group_layout = QFormLayout()
            section_name = "Output"
            for key in self.config[section_name]:
                value = self.config[section_name][key]
                display_key = key.replace("_", " ")
                widget_attr_name = f"{section_name}_{key}_checkbox"
                checkbox = QCheckBox()
                checkbox.setChecked(value.lower() == 'true' or value == '1')
                output_group_layout.addRow(display_key, checkbox)
                setattr(self, widget_attr_name, checkbox)
            output_group_box.setLayout(output_group_layout)
            layout.addWidget(output_group_box)
        layout.addStretch(1)
        self.run_analysis_button = QPushButton("Run Analysis")
        self.run_analysis_button.clicked.connect(self.run_analysis_action)
        layout.addWidget(self.run_analysis_button)
        self.analysis_tab.setLayout(layout)

    def run_analysis_action(self):
        # In background save the current config values in the options tab for consistency
        self.save_config_changes_to_current_file()

        event_sheet_path = self.event_file_path.text()
        file_pair_path = self.file_pair_path.text()

        if not os.path.exists(event_sheet_path) or not os.path.exists(file_pair_path):
            QMessageBox.warning(self, "Error", "Please select valid event and file pair sheets.")
            return

        section_name = "Output"
        output_selections = []
        for i, key in enumerate(self.config[section_name]):
            widget_attr_name = f"{section_name}_{key}_checkbox"
            if hasattr(self, widget_attr_name):
                checkbox = getattr(self, widget_attr_name)
                if checkbox.isChecked():
                    output_selections.append(i + 1)

        # Read num_workers from [Concurrency] section; fall back to 1 (sequential).
        num_workers = 1
        if 'Concurrency' in self.config:
            try:
                num_workers = int(self.config['Concurrency'].get('num_workers', '1'))
            except (ValueError, TypeError):
                num_workers = 1

        # Disable the button while analysis runs to prevent duplicate submissions.
        self.run_analysis_button.setEnabled(False)
        self.run_analysis_button.setText("Running…")

        self._analysis_thread = AnalysisThread(
            file_pair_path, event_sheet_path, output_selections,
            self.config, num_workers, parent=self
        )
        self._analysis_thread.results_ready.connect(self._on_analysis_complete)
        self._analysis_thread.error_occurred.connect(self._on_analysis_error)
        self._analysis_thread.start()

    def _on_analysis_complete(self, results, combined_results):
        """Called on the main thread when the analysis worker finishes."""
        self.analysis_results = results
        self.combined_results = combined_results
        self.run_analysis_button.setEnabled(True)
        self.run_analysis_button.setText("Run Analysis")
        self.update_results_and_visualization_options()
        QMessageBox.information(self, "Analysis", "Analysis complete!")

    def _on_analysis_error(self, error_msg):
        """Called on the main thread when the analysis worker raises an exception."""
        self.run_analysis_button.setEnabled(True)
        self.run_analysis_button.setText("Run Analysis")
        QMessageBox.critical(self, "Analysis Error",
                             f"An error occurred during analysis:\n{error_msg}")

    def update_results_and_visualization_options(self):
        # Refresh animal/date selectors and results table
        self.update_animal_date_selects()
        # update_animal_date_selects triggers cascading date/behavior updates;
        # force a final results-table refresh in case the animal selection did not change.
        self.update_results_table()
        # Ensure the visualization behavior selector is also populated.
        if hasattr(self, 'vis_behavior_select'):
            self.update_vis_behavior_select()

    def init_results_tab(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Analysis Results"))

        controls_layout = QFormLayout()
        # Replace single file selector with Animal ID and Date selectors
        self.results_animal_select = QComboBox()
        self.results_animal_select.currentTextChanged.connect(self.update_results_date_select)
        self.results_date_select = QComboBox()
        self.results_date_select.currentTextChanged.connect(self.update_results_table)
        controls_layout.addRow("Select Animal ID:", self.results_animal_select)
        controls_layout.addRow("Select Date:", self.results_date_select)
        layout.addLayout(controls_layout)

        self.results_table = QTableWidget()
        layout.addWidget(self.results_table)
        save_button = QPushButton("Save Results")
        save_button.clicked.connect(self.save_results)
        layout.addWidget(save_button)
        self.results_tab.setLayout(layout)

    def save_results(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv)")
        if file_path:
            rows = self.results_table.rowCount()
            cols = self.results_table.columnCount()
            data = []
            headers = [self.results_table.horizontalHeaderItem(col).text() for col in range(cols)]
            for row in range(rows):
                row_data = [self.results_table.item(row, col).text() for col in range(cols)]
                data.append(row_data)
            df = pd.DataFrame(data, columns=headers)
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Saved", "Results saved successfully.")

    def init_visualization_tab(self):
        layout = QVBoxLayout()
        
        controls_layout = QFormLayout()
        # Visualization: choose by Animal ID and Date instead of file
        self.vis_animal_select = QComboBox()
        self.vis_date_select = QComboBox()
        # populate selects from any existing results
        self.update_animal_date_selects()
        self.vis_behavior_select = QComboBox()
        self.vis_animal_select.currentTextChanged.connect(self.update_vis_date_select)
        self.vis_date_select.currentTextChanged.connect(self.update_vis_behavior_select)
        
        self.event_prior_input = QLineEdit(self.config['Event_Window']['event_prior'])
        self.event_prior_input.setDisabled(True)
        self.event_follow_input = QLineEdit(self.config['Event_Window']['event_follow'])
        self.event_follow_input.setDisabled(True)
        
        # Visualization mode selector
        self.vis_mode_select = QComboBox()
        self.vis_mode_select.addItems(["Histogram", "Heatmap"])
        controls_layout.addRow("Select Animal ID:", self.vis_animal_select)
        controls_layout.addRow("Select Date:", self.vis_date_select)
        controls_layout.addRow("Select Behavior:", self.vis_behavior_select)
        controls_layout.addRow("Visualization Mode:", self.vis_mode_select)
        
        generate_plot_button = QPushButton("Generate Plot")
        generate_plot_button.clicked.connect(self.generate_plot)
        
        layout.addLayout(controls_layout)
        layout.addWidget(generate_plot_button)
        
        self.canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)
        
        save_layout = QHBoxLayout()
        save_data_button = QPushButton("Save Data")
        save_data_button.clicked.connect(self.save_plot_data)
        save_graph_button = QPushButton("Save Graph")
        save_graph_button.clicked.connect(self.save_plot_graph)
        save_layout.addWidget(save_data_button)
        save_layout.addWidget(save_graph_button)
        
        layout.addLayout(save_layout)
        self.visualization_tab.setLayout(layout)

    def update_animal_date_selects(self):
        """Populate animal and date dropdowns for both Results and Visualization tabs based on analysis_results.
        If a combined (no animal_id) result exists, include the 'Combined' option for animals.
        """
        # Build animals and dates mapping
        animals = set()
        dates_by_animal = {}
        combined_exists = False
        for res in (self.analysis_results or []):
            a = res.get('animal_id')
            d = res.get('date')
            if not a:
                combined_exists = True
                continue
            animals.add(a)
            dates_by_animal.setdefault(a, set()).add(d)

        animals = sorted(animals)

        # Update Results selectors if present
        if hasattr(self, 'results_animal_select') and hasattr(self, 'results_date_select'):
            self.results_animal_select.blockSignals(True)
            self.results_date_select.blockSignals(True)
            self.results_animal_select.clear()
            # Insert Combined at top if exists
            if combined_exists:
                self.results_animal_select.addItem('Combined')
            for a in animals:
                self.results_animal_select.addItem(a)
            # Default date list empty
            self.results_date_select.clear()
            self.results_animal_select.blockSignals(False)
            self.results_date_select.blockSignals(False)
            # Manually trigger date population for the currently selected animal
            # (blockSignals suppresses the currentTextChanged signal, so we call it explicitly)
            current_results_animal = self.results_animal_select.currentText()
            if current_results_animal:
                self.update_results_date_select(current_results_animal)

        # Update Visualization selectors if present
        if hasattr(self, 'vis_animal_select') and hasattr(self, 'vis_date_select'):
            self.vis_animal_select.blockSignals(True)
            self.vis_date_select.blockSignals(True)
            self.vis_animal_select.clear()
            if combined_exists:
                self.vis_animal_select.addItem('Combined')
            for a in animals:
                self.vis_animal_select.addItem(a)
            self.vis_date_select.clear()
            self.vis_animal_select.blockSignals(False)
            self.vis_date_select.blockSignals(False)
            # Manually trigger the vis date + behavior cascade
            current_vis_animal = self.vis_animal_select.currentText()
            if current_vis_animal:
                self.update_vis_date_select(current_vis_animal)

    def update_vis_date_select(self, animal):
        """Populate the visualization date select when the animal selection changes."""
        self.vis_date_select.clear()
        if not animal or not self.analysis_results:
            return
        if animal == 'Combined':
            self.vis_date_select.addItem('Combined')
            # Trigger behavior refresh
            self.update_vis_behavior_select()
            return
        # collect dates for this animal
        dates = sorted({res.get('date') for res in self.analysis_results if res.get('animal_id') == animal and res.get('date')})
        # allow viewing all dates for the animal
        self.vis_date_select.addItem('All')
        for d in dates:
            self.vis_date_select.addItem(d)
        self.update_vis_behavior_select()

    def update_results_date_select(self, animal):
        """Populate the results date select when the results animal selection changes."""
        self.results_date_select.clear()
        if not animal or not self.analysis_results:
            return
        if animal == 'Combined':
            self.results_date_select.addItem('Combined')
            self.update_results_table()
            return
        dates = sorted({res.get('date') for res in self.analysis_results if res.get('animal_id') == animal and res.get('date')})
        self.results_date_select.addItem('All')
        for d in dates:
            self.results_date_select.addItem(d)
        self.update_results_table()

    def update_results_table(self):
        selected_animal = self.results_animal_select.currentText() if hasattr(self, 'results_animal_select') else ''
        selected_date = self.results_date_select.currentText() if hasattr(self, 'results_date_select') else ''
        if not selected_animal or not self.analysis_results:
            self.results_table.clear()
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            return

        # Filter results for the selected animal and date
        def matches_selection(res):
            # Combined entries in analysis_results are the ones missing an animal_id
            res_animal = res.get('animal_id')
            res_date = res.get('date')
            if selected_animal == 'Combined':
                if res_animal:
                    return False
            else:
                if res_animal != selected_animal:
                    return False
            if selected_date and selected_date not in ['Combined', 'All']:
                return res_date == selected_date
            return True

        filtered_results = [res for res in self.analysis_results if matches_selection(res)]

        self.results_table.setRowCount(len(filtered_results))
        self.results_table.setColumnCount(3) # Behavior, Max Peak, AUC
        self.results_table.setHorizontalHeaderLabels(['Behavior', 'Max Peak', 'AUC'])
        for i, res in enumerate(filtered_results):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(res.get('behavior', ''))))
            max_peak = res.get('max_peak')
            auc = res.get('auc')
            self.results_table.setItem(i, 1, QTableWidgetItem(
                f"{float(max_peak):.4f}" if max_peak is not None else 'N/A'))
            self.results_table.setItem(i, 2, QTableWidgetItem(
                f"{float(auc):.4f}" if auc is not None else 'N/A'))
        self.results_table.resizeColumnsToContents()

    def update_vis_file_select(self):
        # legacy helper left for compatibility; delegate to animal/date selector updater
        self.update_animal_date_selects()

    def update_vis_behavior_select(self):
        self.vis_behavior_select.clear()
        selected_animal = self.vis_animal_select.currentText() if hasattr(self, 'vis_animal_select') else ''
        selected_date = self.vis_date_select.currentText() if hasattr(self, 'vis_date_select') else ''
        if not selected_animal or not self.analysis_results:
            return

        def matches(res):
            a = res.get('animal_id')
            d = res.get('date')
            if selected_animal == 'Combined':
                if a:
                    return False
            else:
                if a != selected_animal:
                    return False
            if selected_date and selected_date not in ['All', 'Combined']:
                return d == selected_date
            return True

        behaviors = sorted(list(set([res['behavior'] for res in self.analysis_results if matches(res)])))
        self.vis_behavior_select.addItems(behaviors)

    def generate_plot(self):
        selected_animal = self.vis_animal_select.currentText() if hasattr(self, 'vis_animal_select') else ''
        selected_date = self.vis_date_select.currentText() if hasattr(self, 'vis_date_select') else ''
        behavior = self.vis_behavior_select.currentText()
        event_prior = float(self.event_prior_input.text())
        event_follow = float(self.event_follow_input.text())
        vis_mode = self.vis_mode_select.currentText().lower() if hasattr(self, 'vis_mode_select') else 'histogram'

        # Find the data in self.analysis_results that matches the selected animal/date and behavior
        def matches(res):
            a = res.get('animal_id')
            d = res.get('date')
            if selected_animal == 'Combined':
                if a:
                    return False
            else:
                if a != selected_animal:
                    return False
            if selected_date and selected_date not in ['All', 'Combined']:
                return d == selected_date
            return res.get('behavior') == behavior

        result_data = next((res for res in self.analysis_results if matches(res) and res.get('behavior') == behavior), None)

        if result_data and not result_data['plot_data'].empty:
            self.plot_data = result_data['plot_data']
        else:
            self.canvas.axes.clear()
            self.canvas.draw()
            QMessageBox.warning(self, "Data Not Found", f"No data matches the selected file and behavior, or the data is empty.")
            return

        # Clear the canvas and re-add axes before plotting
        self.canvas.figure.clf()
        self.canvas.axes = self.canvas.figure.add_subplot(111)
        time_axis = np.linspace(-event_prior, event_follow, len(self.plot_data.index))

        if vis_mode == 'histogram':
            mean_data = self.plot_data.mean(axis=1)
            sem_data = self.plot_data.sem(axis=1)
            self.canvas.axes.plot(time_axis, mean_data, label='Mean')
            self.canvas.axes.fill_between(time_axis, mean_data - sem_data, mean_data + sem_data, alpha=0.2, label='SEM')
            self.canvas.axes.axvline(x=0, color='r', linestyle='--')
            self.canvas.axes.set_xlabel("Time (s)")
            self.canvas.axes.set_ylabel("Signal")
            self.canvas.axes.set_title(f"Perievent Histogram for {behavior}")
            self.canvas.axes.legend()
        elif vis_mode == 'heatmap':
            # Each event instance is a row, first event at the top
            data = self.plot_data.T.values  # shape: (n_events, n_timepoints)
            im = self.canvas.axes.imshow(data, aspect='auto', cmap='viridis',
                                         extent=[-event_prior, event_follow, data.shape[0], 0])
            self.canvas.axes.axvline(x=0, color='r', linestyle='--')
            self.canvas.axes.set_xlabel("Time (s)")
            self.canvas.axes.set_ylabel("Event Instance")
            self.canvas.axes.set_title(f"Perievent Heatmap for {behavior}")
            cbar = self.canvas.figure.colorbar(im, ax=self.canvas.axes)
            cbar.set_label("Signal")
        self.canvas.draw()

    def save_plot_data(self):
        if not hasattr(self, 'plot_data') or self.plot_data.empty:
            QMessageBox.warning(self, "No Data", "Generate a plot first before saving plot data.")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot Data", "", "CSV Files (*.csv)")
        if file_path:
            self.plot_data.to_csv(file_path)
            QMessageBox.information(self, "Saved", "Plot data saved successfully.")

    def save_plot_graph(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Graph", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_path:
            self.canvas.figure.savefig(file_path)
            QMessageBox.information(self, "Saved", "Graph saved successfully.")

    def init_options_tab(self):
        # Create a scrollable area for the Options tab
        outer_layout = QVBoxLayout()
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        multibox_options = ['trial_start_stage', 'trial_end_stage', 'exclusion_list']
        file_path_keys = {'file_list_path': 'open', 'event_list_path': 'open', 'output_path': 'directory'}
        for section in self.config.sections():
            if section == "Output":
                continue
            group_box = QGroupBox(section.replace("_", " "))
            group_layout = QFormLayout()
            fit_type_combo = None
            arpls_widgets = []
            # First pass: create fit_type combo if present
            for key in self.config[section]:
                if key == 'fit_type':
                    value = self.config[section][key]
                    display_key = key.replace("_", " ")
                    fit_type_combo = QComboBox()
                    fit_type_combo.addItems(['linear', 'expodecay', 'arpls'])
                    fit_type_combo.setCurrentText(value)
                    group_layout.addRow(display_key, fit_type_combo)
                    setattr(self, f"{section}_fit_type_combobox", fit_type_combo)
            # Second pass: all other keys
            for key in self.config[section]:
                if key == 'fit_type':
                    continue
                value = self.config[section][key]
                display_key = key.replace("_", " ")
                widget_attr_base = f"{section}_{key}"
                # Multi-select options use the custom widget
                if key in multibox_options:
                    multicombobox_edit = MultiSelectComboBox()
                    setattr(self, f"{widget_attr_base}_multicombobox_edit", multicombobox_edit)
                    if value:
                        config_items = [item.strip() for item in value.split(',') if item.strip()]
                        for item_text in config_items:
                            multicombobox_edit.add_option(item_text)
                            for i in range(multicombobox_edit.list_widget.count()):
                                list_item = multicombobox_edit.list_widget.item(i)
                                if list_item.text() == item_text:
                                    list_item.setCheckState(Qt.Checked)
                                    break
                    group_layout.addRow(display_key, multicombobox_edit)
                # If the key is one of the file path keys, create a line edit + browse button
                elif key in file_path_keys:
                    hbox = QHBoxLayout()
                    line_edit = QLineEdit(value)
                    browse_btn = QPushButton("Browse...")

                    def make_browse_handler(le=line_edit, k=key):
                        def handler():
                            mode = file_path_keys.get(k, 'open')
                            if mode == 'open':
                                file_path, _ = QFileDialog.getOpenFileName(self, f"Select {display_key}", "", "All Files (*)")
                                if file_path:
                                    le.setText(file_path)
                            elif mode == 'directory':
                                dir_path = QFileDialog.getExistingDirectory(self, f"Select {display_key}")
                                if dir_path:
                                    le.setText(dir_path)
                            else:
                                # default to open file
                                file_path, _ = QFileDialog.getOpenFileName(self, f"Select {display_key}", "", "All Files (*)")
                                if file_path:
                                    le.setText(file_path)
                        return handler

                    browse_btn.clicked.connect(make_browse_handler())
                    hbox.addWidget(line_edit)
                    hbox.addWidget(browse_btn)
                    setattr(self, f"{widget_attr_base}_line_edit", line_edit)
                    group_layout.addRow(display_key, hbox)
                # Boolean-ish values (except in Output section) get checkboxes
                elif value.lower() in ['true', 'false', '1', '0'] and section != "Output":
                    checkbox = QCheckBox()
                    checkbox.setChecked(value.lower() == 'true' or value == '1')
                    group_layout.addRow(display_key, checkbox)
                    setattr(self, f"{widget_attr_base}_checkbox", checkbox)
                elif key == 'filter_type':
                    combo = QComboBox()
                    combo.addItems(['lowpass', 'smoothing'])
                    combo.setCurrentText(value)
                    group_layout.addRow(display_key, combo)
                    setattr(self, f"{widget_attr_base}_combobox", combo)
                elif key == 'filter_name':
                    combo = QComboBox()
                    # Find the corresponding filter_type combobox
                    filter_type_key = f"{section}_filter_type_combobox"
                    filter_type_combo = getattr(self, filter_type_key, None)
                    def update_filter_name_options():
                        filter_type_val = filter_type_combo.currentText() if filter_type_combo else self.config[section].get('filter_type', 'lowpass')
                        combo.clear()
                        if filter_type_val == 'lowpass':
                            combo.addItems(['butter', 'bessel', 'chebychev'])
                        elif filter_type_val == 'smoothing':
                            combo.addItem('savitsky-golay')
                        # Set to config value if present
                        combo.setCurrentText(value)
                    # If filter_type combobox exists, connect signal
                    if filter_type_combo:
                        filter_type_combo.currentTextChanged.connect(lambda _: update_filter_name_options())
                    update_filter_name_options()
                    group_layout.addRow(display_key, combo)
                    setattr(self, f"{widget_attr_base}_combobox", combo)
                elif key == 'savgol_window' or key == 'savgol_polyorder':
                    # Only show if filter_name combobox is set to 'savitsky-golay'
                    filter_name_key = f"{section}_filter_name_combobox"
                    filter_name_combo = getattr(self, filter_name_key, None)
                    spin_box = QSpinBox()
                    spin_box.setValue(int(value))
                    label = QLabel(display_key)
                    group_layout.addRow(label, spin_box)
                    spin_box.setVisible(False)
                    label.setVisible(False)
                    setattr(self, f"{widget_attr_base}_spin_box", spin_box)
                    def update_savgol_visibility():
                        visible = filter_name_combo and filter_name_combo.currentText() == 'savitsky-golay'
                        spin_box.setVisible(visible)
                        label.setVisible(visible)
                    if filter_name_combo:
                        filter_name_combo.currentTextChanged.connect(lambda _: update_savgol_visibility())
                        update_savgol_visibility()
                elif key == 'filter_order' or key == 'filter_cutoff':
                    # Only show if filter_type combobox is set to 'lowpass'
                    filter_type_key = f"{section}_filter_type_combobox"
                    filter_type_combo = getattr(self, filter_type_key, None)
                    spin_box = QSpinBox()
                    config_default = self.config[section].get(key, None)
                    if key == 'filter_cutoff':
                        spin_box.setRange(1, 1000)
                        default_value = int(config_default) if config_default and config_default.isdigit() else 1
                    else:
                        spin_box.setRange(1, 10)
                        default_value = int(config_default) if config_default and config_default.isdigit() else 1
                    spin_box.setValue(default_value)
                    label = QLabel(display_key)
                    group_layout.addRow(label, spin_box)
                    spin_box.setVisible(False)
                    label.setVisible(False)
                    setattr(self, f"{widget_attr_base}_spin_box", spin_box)
                    def update_lowpass_visibility():
                        visible = filter_type_combo and filter_type_combo.currentText() == 'lowpass'
                        spin_box.setVisible(visible)
                        label.setVisible(visible)
                    if filter_type_combo:
                        filter_type_combo.currentTextChanged.connect(lambda _: update_lowpass_visibility())
                        update_lowpass_visibility()
                elif key == 'cheby_ripple':
                    # Only show if filter_name combobox is 'chebychev' AND filter_type combobox is 'smoothing'
                    filter_name_key = f"{section}_filter_name_combobox"
                    filter_type_key = f"{section}_filter_type_combobox"
                    filter_name_combo = getattr(self, filter_name_key, None)
                    filter_type_combo = getattr(self, filter_type_key, None)
                    double_spin_box = QSpinBox()
                    double_spin_box.setRange(1, 10)
                    double_spin_box.setValue(int(value) if value.isdigit() else 1)
                    label = QLabel(display_key)
                    group_layout.addRow(label, double_spin_box)
                    double_spin_box.setVisible(False)
                    label.setVisible(False)
                    setattr(self, f"{widget_attr_base}_double_spin_box", double_spin_box)
                    def update_cheby_visibility():
                        visible = (
                            filter_name_combo and filter_type_combo and
                            filter_name_combo.currentText() == 'chebychev' and
                            filter_type_combo.currentText() == 'smoothing'
                        )
                        double_spin_box.setVisible(visible)
                        label.setVisible(visible)
                    if filter_name_combo:
                        filter_name_combo.currentTextChanged.connect(lambda _: update_cheby_visibility())
                    if filter_type_combo:
                        filter_type_combo.currentTextChanged.connect(lambda _: update_cheby_visibility())
                    update_cheby_visibility()
                # ARPLS options: only show if fit_type is 'arpls'
                elif key.startswith('arpls_'):
                    arpls_widget = QLineEdit(value)
                    setattr(self, f"{widget_attr_base}_line_edit", arpls_widget)
                    label = QLabel(display_key)
                    group_layout.addRow(label, arpls_widget)
                    arpls_widgets.append((label, arpls_widget))
                else:
                    line_edit = QLineEdit(value)
                    setattr(self, f"{widget_attr_base}_line_edit", line_edit)
                    group_layout.addRow(display_key, line_edit)
            # If fit_type_combo and arpls_widgets exist, connect visibility
            if fit_type_combo and arpls_widgets:
                def update_arpls_visibility():
                    if fit_type_combo is not None:
                        is_arpls = fit_type_combo.currentText() == 'arpls'
                        for label, widget in arpls_widgets:
                            label.setVisible(is_arpls)
                            widget.setVisible(is_arpls)
                if fit_type_combo is not None:
                    fit_type_combo.currentTextChanged.connect(lambda _: update_arpls_visibility())
                    update_arpls_visibility()
            group_box.setLayout(group_layout)
            content_layout.addWidget(group_box)
        save_button = QPushButton("Save Changes")
        save_button.clicked.connect(self.save_config_changes_to_current_file)
        content_layout.addWidget(save_button)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        outer_layout.addWidget(scroll)
        self.options_tab.setLayout(outer_layout)

    def update_options_with_template(self):
        if self.template_loaded:
            widgets_to_update = [
                'ITI_Window_trial_start_stage_multicombobox_edit',
                'ITI_Window_trial_end_stage_multicombobox_edit',
                'Filter_exclusion_list_multicombobox_edit'
            ]
            for stage in self.trial_stage_options:
                for widget_name in widgets_to_update:
                    if hasattr(self, widget_name):
                        widget = getattr(self, widget_name)
                        if isinstance(widget, MultiSelectComboBox):
                            widget.add_option(stage)
                    else:
                        print(f"Warning: Widget {widget_name} not found while updating options with template.")
        else:
            QMessageBox.warning(self, "Error", "No template loaded, cannot update trial start stage.")

    def update_ui_from_config(self):
        multibox_options = ['trial_start_stage', 'trial_end_stage', 'exclusion_list']
        for section in self.config.sections():
            for key in self.config[section]:
                value = self.config[section][key]
                widget_attr_base = f"{section}_{key}"
                if hasattr(self, f"{widget_attr_base}_checkbox"):
                    widget = getattr(self, f"{widget_attr_base}_checkbox")
                    widget.setChecked(value.lower() == 'true' or value == '1')
                elif hasattr(self, f"{widget_attr_base}_multicombobox_edit"):
                    widget = getattr(self, f"{widget_attr_base}_multicombobox_edit")
                    config_values_list = [v.strip() for v in value.split(',') if v.strip()]
                    for i in range(widget.list_widget.count()):
                        widget.list_widget.item(i).setCheckState(Qt.Unchecked)
                    for val_from_config in config_values_list:
                        found = any(widget.list_widget.item(i).text() == val_from_config for i in range(widget.list_widget.count()))
                        if not found:
                            widget.add_option(val_from_config)
                        for i in range(widget.list_widget.count()):
                            item = widget.list_widget.item(i)
                            if item.text() == val_from_config:
                                item.setCheckState(Qt.Checked)
                                break
                elif hasattr(self, f"{widget_attr_base}_line_edit"):
                    widget = getattr(self, f"{widget_attr_base}_line_edit")
                    widget.setText(value)
        if self.template_loaded:
            self.update_options_with_template()

    def save_config_changes_to_current_file(self):
        # Sync UI widgets into the in-memory config, then write to file
        self.update_config_from_ui()
        try:
            with open(self.config_file_path, 'w', encoding='utf-8') as configfile:
                self.config.write(configfile)
            # Update snapshot so subsequent close checks know this is the saved state
            try:
                import io
                buf = io.StringIO()
                self.config.write(buf)
                self._original_config_text = buf.getvalue()
            except Exception:
                self._original_config_text = None
            QMessageBox.information(self, "Saved", f"Configuration changes saved successfully to {self.config_file_path}.")
        except IOError as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save configuration to {self.config_file_path}: {e}")

    def update_config_from_ui(self):
        """Read widgets created in the Options tab and write their values into self.config (in-memory).
        This centralizes UI->config mapping so callers can update before writing to disk.
        """
        for section in self.config.sections():
            for key in self.config[section]:
                widget_attr_base = f"{section}_{key}"
                # try common widget suffixes used in the options creation
                widget = getattr(self, f"{widget_attr_base}_line_edit", None) or \
                         getattr(self, f"{widget_attr_base}_checkbox", None) or \
                         getattr(self, f"{widget_attr_base}_multicombobox_edit", None) or \
                         getattr(self, f"{widget_attr_base}_combobox", None) or \
                         getattr(self, f"{widget_attr_base}_spin_box", None) or \
                         getattr(self, f"{widget_attr_base}_double_spin_box", None)
                # Map widget types back to string values for config
                if widget is None:
                    continue
                try:
                    # QLineEdit
                    if isinstance(widget, QLineEdit):
                        self.config[section][key] = widget.text()
                    # QCheckBox
                    elif isinstance(widget, QCheckBox):
                        self.config[section][key] = 'true' if widget.isChecked() else 'false'
                    # MultiSelectComboBox
                    elif isinstance(widget, MultiSelectComboBox):
                        checked_items = widget.get_checked_items()
                        self.config[section][key] = ",".join(checked_items)
                    # QComboBox
                    elif isinstance(widget, QComboBox):
                        self.config[section][key] = widget.currentText()
                    # QSpinBox (including the integer spinboxes used for various numeric options)
                    elif isinstance(widget, QSpinBox):
                        self.config[section][key] = str(widget.value())
                    else:
                        # Fallback: try to get a sensible string value
                        val = None
                        if hasattr(widget, 'text'):
                            try:
                                val = widget.text()
                            except Exception:
                                pass
                        if val is None and hasattr(widget, 'currentText'):
                            try:
                                val = widget.currentText()
                            except Exception:
                                pass
                        if val is not None:
                            self.config[section][key] = str(val)
                except Exception:
                    # If a single widget fails to serialize, skip it but don't abort the whole update
                    continue

    def load_configuration_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Configuration File", "", "INI Files (*.ini)")
        if file_path:
            self.config_file_path = file_path
            self.config = configparser.ConfigParser()
            try:
                if not self.config.read(self.config_file_path):
                    QMessageBox.warning(self, "Load Warning", f"Configuration file {self.config_file_path} not found or empty.")
                    return
                self.update_ui_from_config()
                # After successfully loading a new configuration, snapshot it as the original
                try:
                    with open(self.config_file_path, 'r', encoding='utf-8') as f:
                        self._original_config_text = f.read()
                except Exception:
                    import io
                    buf = io.StringIO()
                    self.config.write(buf)
                    self._original_config_text = buf.getvalue()
                QMessageBox.information(self, "Loaded", f"Configuration loaded from: {self.config_file_path}")
            except configparser.Error as e:
                QMessageBox.critical(self, "Load Error", f"Error reading configuration file {self.config_file_path}: {e}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"An unexpected error occurred while loading configuration: {e}")

    def save_configuration_as(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Configuration As...", "", "INI Files (*.ini)")
        if file_path:
            self.config_file_path = file_path
            self.save_config_changes_to_current_file()

    def load_defaults_from_config(self):
        """Load default event and file-pair sheets from the [Filepath] section in the config.
        Maps:
          - event_list_path -> event sheet (self.event_table)
          - file_list_path  -> file pair sheet (self.file_pair_table)
        This will set the QLineEdit paths and call display_csv_in_table to populate the tables.
        """
        try:
            if "Filepath" not in self.config:
                return

            file_list_path = self.config['Filepath'].get('file_list_path', '').strip()
            event_list_path = self.config['Filepath'].get('event_list_path', '').strip()

            # Load event sheet if provided and is a file
            if event_list_path:
                try:
                    if os.path.isfile(event_list_path):
                        self.event_file_path.setText(event_list_path)
                        self.display_csv_in_table(event_list_path, self.event_table)
                    else:
                        # If path isn't a file, don't attempt to load but still set the path text
                        self.event_file_path.setText(event_list_path)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load event sheet from config: {e}")

            # Load file pair sheet if provided and is a file
            if file_list_path:
                try:
                    if os.path.isfile(file_list_path):
                        self.file_pair_path.setText(file_list_path)
                        self.display_csv_in_table(file_list_path, self.file_pair_table)
                    else:
                        # If path isn't a file, don't attempt to load but still set the path text
                        self.file_pair_path.setText(file_list_path)
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load file pair sheet from config: {e}")
        except Exception as e:
            # Protect initialization from crashing if config is malformed
            print(f"Warning: load_defaults_from_config failed: {e}")

    def _get_config_text(self):
        import io
        buf = io.StringIO()
        self.config.write(buf)
        return buf.getvalue()

    def closeEvent(self, event):
        """Prompt to save config if it has changed since it was loaded/snapshotted."""
        try:
            # First ensure UI values are reflected in the in-memory config
            try:
                self.update_config_from_ui()
            except Exception:
                # If updating from UI fails, fall back to comparing current parser state
                pass

            current_text = self._get_config_text() if hasattr(self, '_get_config_text') else None
            original_text = self._original_config_text if hasattr(self, '_original_config_text') else None
            if original_text is None:
                changed = bool(current_text and current_text.strip())
            else:
                changed = (current_text != original_text)
        except Exception:
            changed = False

        if changed:
            # Auto-save changes without prompting, but allow cancel if save fails
            try:
                with open(self.config_file_path, 'w', encoding='utf-8') as configfile:
                    self.config.write(configfile)
                # Update snapshot so subsequent checks are accurate
                try:
                    import io
                    buf = io.StringIO()
                    self.config.write(buf)
                    self._original_config_text = buf.getvalue()
                except Exception:
                    self._original_config_text = None
            except Exception as e:
                # If saving fails, give the user a chance to cancel close
                reply = QMessageBox.question(self, "Save Failed",
                                             f"Failed to save configuration to {self.config_file_path}: {e}\nDo you want to cancel closing and try to save manually?",
                                             QMessageBox.Yes | QMessageBox.No,
                                             QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    event.ignore()
                    return
                # If user chooses No, proceed to close without saved changes
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = FiberPhotometryApp()
    main_win.show()
    sys.exit(app.exec())