import sys
import pandas as pd
import os
import configparser
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                               QPushButton, QLabel, QFileDialog, QLineEdit, QTableWidget,
                               QTableWidgetItem, QFormLayout, QMessageBox,
                               QGroupBox, QCheckBox, QHBoxLayout, QMenuBar, QComboBox,
                               QListWidget, QListWidgetItem, QSpinBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from functools import partial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from Processing import data_processor
import multiprocessing

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

class FiberPhotometryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fiber Photometry Data Analyzer")
        self.resize(1200, 800)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file_path = os.path.normpath(os.path.join(script_dir, '..', 'Config.ini'))
        self.config = configparser.ConfigParser()

        self.results_data = []

        try:
            if not self.config.read(self.config_file_path):
                QMessageBox.warning(self, "Config File Warning", f"Config file '{self.config_file_path}' not found or empty. Defaults may not be loaded correctly.")
        except configparser.Error as e:
            QMessageBox.critical(self, "Config File Error", f"Error reading config file '{self.config_file_path}': {e}")

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
                else:
                    item = QTableWidgetItem(str(data.iat[row, col]))
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    table_widget.setItem(row, col, item)
        table_widget.resizeColumnsToContents()

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
                    if isinstance(table_widget.cellWidget(row, col), QComboBox):
                        row_data.append(table_widget.cellWidget(row, col).currentText())
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
        event_sheet_path = self.event_file_path.text()        
        print(event_sheet_path)
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
                    output_selections.append(i+1)

        self.results_data = data_processor.process_files(file_pair_path, event_sheet_path, output_selections, self.config)

        # Display the results
        self.display_results(self.results_data)
        QMessageBox.information(self, "Analysis", "Analysis complete!")

    def display_results(self, results_data):
        self.results_table.setRowCount(len(results_data))
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(['File', 'Behavior', 'Max Peak', 'AUC'])
        for i, row_data in enumerate(results_data):
            for j, cell_data in enumerate(row_data[:4]):
                self.results_table.setItem(i, j, QTableWidgetItem(str(cell_data)))
        self.results_table.resizeColumnsToContents()

    def init_results_tab(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Analysis Results"))
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
        self.vis_file_select = QComboBox()
        self.update_vis_file_select()
        self.vis_behavior_select = QComboBox()
        self.vis_file_select.currentTextChanged.connect(self.update_vis_behavior_select)
        
        self.event_prior_input = QLineEdit("2.0")
        self.event_follow_input = QLineEdit("5.0")
        
        controls_layout.addRow("Select File:", self.vis_file_select)
        controls_layout.addRow("Select Behavior:", self.vis_behavior_select)
        controls_layout.addRow("Event Prior (s):", self.event_prior_input)
        controls_layout.addRow("Event Follow (s):", self.event_follow_input)
        
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

    def update_vis_file_select(self):
        self.vis_file_select.clear()
        for i, row_data in enumerate(self.results_data):
            self.vis_file_select.addItems(row_data[0])

    def update_vis_behavior_select(self):
        self.vis_behavior_select.clear()
        for i, row_data in enumerate(self.results_data):
            self.vis_file_select.addItems(row_data[1])

    def generate_plot(self):
        file_path = self.vis_file_select.currentText()
        behavior = self.vis_behavior_select.currentText()
        event_prior = float(self.event_prior_input.text())
        event_follow = float(self.event_follow_input.text())

        # Find the index in self.results_data that matches the file_path and behavior
        found_index = next((i for i, row in enumerate(self.results_data) if row[0] == file_path and row[1] == behavior), None)

        if found_index is not None:
            self.plot_data = self.results_data[found_index][5]  # plot data is in index 2
        else:
            QMessageBox.warning(self, "Data Not Found", f"No data matches the selected file and behavior.")
            return
        
        if not self.plot_data.empty:
            self.canvas.axes.clear()
            time_axis = np.linspace(-event_prior, event_follow, len(self.plot_data))
            mean_data = self.plot_data.mean(axis=1)
            sem_data = self.plot_data.sem(axis=1)
            self.canvas.axes.plot(time_axis, mean_data, label='Mean')
            self.canvas.axes.fill_between(time_axis, mean_data - sem_data, mean_data + sem_data, alpha=0.2, label='SEM')
            self.canvas.axes.axvline(x=0, color='r', linestyle='--')
            self.canvas.axes.set_xlabel("Time (s)")
            self.canvas.axes.set_ylabel("Signal")
            self.canvas.axes.set_title(f"Perievent Histogram for {behavior}")
            self.canvas.draw()

    def save_plot_data(self):
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
        main_layout = QVBoxLayout()
        multibox_options = ['trial_start_stage', 'trial_end_stage', 'exclusion_list']
        for section in self.config.sections():
            if section == "Output":
                continue
            group_box = QGroupBox(section.replace("_", " "))
            group_layout = QFormLayout()
            for key in self.config[section]:
                value = self.config[section][key]
                display_key = key.replace("_", " ")
                widget_attr_base = f"{section}_{key}"
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
                elif value.lower() in ['true', 'false', '1', '0'] and section != "Output":
                    checkbox = QCheckBox()
                    checkbox.setChecked(value.lower() == 'true' or value == '1')
                    group_layout.addRow(display_key, checkbox)
                else:
                    line_edit = QLineEdit(value)
                    setattr(self, f"{widget_attr_base}_line_edit", line_edit)
                    group_layout.addRow(display_key, line_edit)
            group_box.setLayout(group_layout)
            main_layout.addWidget(group_box)
        save_button = QPushButton("Save Changes")
        save_button.clicked.connect(self.save_config_changes_to_current_file)
        main_layout.addWidget(save_button)
        self.options_tab.setLayout(main_layout)

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
        for section in self.config.sections():
            for key in self.config[section]:
                widget_attr_base = f"{section}_{key}"
                widget = getattr(self, f"{widget_attr_base}_line_edit", None) or \
                         getattr(self, f"{widget_attr_base}_checkbox", None) or \
                         getattr(self, f"{widget_attr_base}_multicombobox_edit", None)
                if isinstance(widget, QLineEdit):
                    self.config[section][key] = widget.text()
                elif isinstance(widget, QCheckBox):
                    self.config[section][key] = 'true' if widget.isChecked() else 'false'
                elif isinstance(widget, MultiSelectComboBox):
                    checked_items = widget.get_checked_items()
                    self.config[section][key] = ",".join(checked_items)
        try:
            with open(self.config_file_path, 'w') as configfile:
                self.config.write(configfile)
            QMessageBox.information(self, "Saved", f"Configuration changes saved successfully to {self.config_file_path}.")
        except IOError as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save configuration to {self.config_file_path}: {e}")

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

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    main_win = FiberPhotometryApp()
    main_win.show()
    sys.exit(app.exec())