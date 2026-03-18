import sys
import pandas as pd
import os
import configparser
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                               QPushButton, QLabel, QFileDialog, QLineEdit, QTableWidget,
                               QTableWidgetItem, QFormLayout, QMessageBox,
                               QGroupBox, QCheckBox, QHBoxLayout, QMenuBar, QComboBox,
                               QListWidget, QListWidgetItem, QSpinBox, QScrollArea,
                               QStackedWidget, QGridLayout, QProgressBar, QStatusBar, QMenu)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QAction, QIcon, QFont, QCursor
from functools import partial

# type: ignore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from Processing import data_processor
from PySide6.QtWidgets import QDialog

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.tight_layout()
        super(MatplotlibCanvas, self).__init__(fig)

class PhotometryPreviewWindow(QDialog):
    def __init__(self, config, file_pair_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Photometry Preview Window")
        self.resize(1000, 800)
        self.config = config
        self.file_pair_data = file_pair_data
        
        main_layout = QHBoxLayout(self)
        
        # Left side controls
        controls_layout = QVBoxLayout()
        controls_form = QFormLayout()
        
        self.file_select = QComboBox()
        self.populate_file_options()
        controls_form.addRow("Select Doric File:", self.file_select)
        
        self.filter_type = QComboBox()
        self.filter_type.addItems(['lowpass', 'smoothing'])
        self.filter_type.setCurrentText(self.config['Photometry_Processing'].get('filter_type', 'lowpass'))
        controls_form.addRow("Filter Type:", self.filter_type)
        
        self.filter_name = QComboBox()
        self.update_filter_name_options()
        self.filter_name.setCurrentText(self.config['Photometry_Processing'].get('filter_name', 'butter'))
        controls_form.addRow("Filter Name:", self.filter_name)
        
        self.filter_type.currentTextChanged.connect(self.update_filter_name_options)
        
        self.filter_order = QSpinBox()
        self.filter_order.setRange(1, 10)
        self.filter_order.setValue(int(self.config['Photometry_Processing'].get('filter_order', '4')))
        controls_form.addRow("Filter/Savgol Order:", self.filter_order)
        
        self.filter_cutoff = QSpinBox()
        self.filter_cutoff.setRange(1, 1000)
        self.filter_cutoff.setValue(int(self.config['Photometry_Processing'].get('filter_cutoff', '10')))
        controls_form.addRow("Cutoff/Savgol Window:", self.filter_cutoff)
        
        self.crop_start = QLineEdit(self.config['Photometry_Processing'].get('crop_start', '0.0'))
        controls_form.addRow("Crop Start (s):", self.crop_start)
        
        self.crop_end = QLineEdit(self.config['Photometry_Processing'].get('crop_end', '0.0'))
        controls_form.addRow("Crop End (s):", self.crop_end)
        
        self.fit_type = QComboBox()
        self.fit_type.addItems(['linear', 'expodecay', 'arpls'])
        self.fit_type.setCurrentText(self.config['Photometry_Processing'].get('fit_type', 'linear'))
        controls_form.addRow("Fit Type:", self.fit_type)
        
        self.arpls_lambda = QLineEdit(self.config['Photometry_Processing'].get('arpls_lambda', '1e5'))
        controls_form.addRow("arPLS Lambda:", self.arpls_lambda)
        
        controls_layout.addLayout(controls_form)
        
        self.update_btn = QPushButton("Update Preview")
        self.update_btn.clicked.connect(self.update_preview)
        controls_layout.addWidget(self.update_btn)
        controls_layout.addStretch()
        
        # Right side plots
        plots_layout = QVBoxLayout()
        self.raw_canvas = MatplotlibCanvas(self, width=6, height=3)
        self.filtered_canvas = MatplotlibCanvas(self, width=6, height=3)
        self.fitted_canvas = MatplotlibCanvas(self, width=6, height=3)
        
        plots_layout.addWidget(QLabel("Raw Data (Isobestic and Active)"))
        plots_layout.addWidget(self.raw_canvas)
        plots_layout.addWidget(QLabel("Filtered Data"))
        plots_layout.addWidget(self.filtered_canvas)
        plots_layout.addWidget(QLabel("Filtered & Fitted Data (Delta F/F)"))
        plots_layout.addWidget(self.fitted_canvas)
        
        main_layout.addLayout(controls_layout, 1)
        main_layout.addLayout(plots_layout, 3)
        
    def populate_file_options(self):
        if self.file_pair_data is not None and not self.file_pair_data.empty:
            for index, row in self.file_pair_data.iterrows():
                doric_path = row.get('doric_path')
                if pd.notna(doric_path):
                    self.file_select.addItem(str(doric_path), userData=row.to_dict())
                    
    def update_filter_name_options(self):
        self.filter_name.clear()
        if self.filter_type.currentText() == 'lowpass':
            self.filter_name.addItems(['butter', 'bessel', 'chebychev'])
        elif self.filter_type.currentText() == 'smoothing':
            self.filter_name.addItem('savitsky-golay')
            
    def update_preview(self):
        import traceback
        try:
            row_dict = self.file_select.currentData()
            if not row_dict:
                QMessageBox.warning(self, "No file selected", "Please select a valid doric file from the dropdown.")
                return
            
            self.update_btn.setText("Updating...")
            self.update_btn.setEnabled(False)
            QApplication.processEvents()
            
            photometry_data = data_processor.PhotometryData()
            
            # Optionally load ABET to synchronize if available
            abet_path = row_dict.get('abet_path')
            if pd.notna(abet_path) and os.path.exists(abet_path):
                 photometry_data.load_abet_data(abet_path)
            
            doric_path = row_dict.get('doric_path')
            if not pd.notna(doric_path) or not os.path.exists(doric_path):
                 QMessageBox.warning(self, "File Error", "Doric path is invalid or file does not exist.")
                 self.reset_button()
                 return
                 
            photometry_data.load_doric_data(
                doric_path,
                row_dict.get('ctrl_col_num'),
                row_dict.get('act_col_num'),
                row_dict.get('ttl_col_num'),
                row_dict.get('mode')
            )
            
            if photometry_data.abet_loaded:
                photometry_data.abet_doric_synchronize()
                
            try:
                crop_start_val = float(self.crop_start.text())
            except ValueError:
                crop_start_val = 0.0
                
            try:
                crop_end_val = float(self.crop_end.text())
            except ValueError:
                crop_end_val = 0.0
                
            photometry_data.doric_crop(start_time_remove=crop_start_val, end_time_remove=crop_end_val)
            
            # Plot Raw
            self.raw_canvas.axes.clear()
            if not photometry_data.doric_pandas.empty:
                time_raw = photometry_data.doric_pandas['Time']
                self.raw_canvas.axes.plot(time_raw, photometry_data.doric_pandas['Control'], label='Control (Isobestic)', alpha=0.7)
                self.raw_canvas.axes.plot(time_raw, photometry_data.doric_pandas['Active'], label='Active', alpha=0.7)
                self.raw_canvas.axes.legend()
                self.raw_canvas.axes.set_xlabel('Time (s)')
                self.raw_canvas.axes.set_ylabel('Fluorescence')
            self.raw_canvas.draw()
            
            # Filter
            time_data, filtered_f0, filtered_f = photometry_data.doric_filter(
                filter_type=self.filter_type.currentText(),
                filter_name=self.filter_name.currentText(),
                filter_order=self.filter_order.value(),
                filter_cutoff=self.filter_cutoff.value()
            )
            
            # Plot Filtered
            self.filtered_canvas.axes.clear()
            self.filtered_canvas.axes.plot(time_data, filtered_f0, label='Filtered Control', alpha=0.8)
            self.filtered_canvas.axes.plot(time_data, filtered_f, label='Filtered Active', alpha=0.8)
            self.filtered_canvas.axes.legend()
            self.filtered_canvas.axes.set_xlabel('Time (s)')
            self.filtered_canvas.axes.set_ylabel('Fluorescence')
            self.filtered_canvas.draw()
            
            # Fit
            # Temporarily set arpls lambda if arpls
            orig_lambda = photometry_data.config['Photometry_Processing'].get('arpls_lambda') if hasattr(photometry_data, 'config') else None
            if not hasattr(photometry_data, 'config'):
                photometry_data.config = {'Photometry_Processing': {}}
            photometry_data.config['Photometry_Processing']['arpls_lambda'] = self.arpls_lambda.text()
            
            photometry_data.doric_fit(self.fit_type.currentText(), filtered_f0, filtered_f, time_data)
            
            # Plot Fitted
            self.fitted_canvas.axes.clear()
            if not photometry_data.doric_pd.empty:
                time_fit = photometry_data.doric_pd['Time']
                deltaf = photometry_data.doric_pd['DeltaF']
                self.fitted_canvas.axes.plot(time_fit, deltaf, color='green', label='Delta F/F')
                self.fitted_canvas.axes.legend()
                self.fitted_canvas.axes.set_xlabel('Time (s)')
                self.fitted_canvas.axes.set_ylabel('Delta F/F')
            self.fitted_canvas.draw()
            
            self.reset_button()
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Preview Error", f"An error occurred generating the preview:\n{str(e)}")
            self.reset_button()
            
    def reset_button(self):
        self.update_btn.setText("Update Preview")
        self.update_btn.setEnabled(True)

class MultiSelectComboBox(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Select or add new entries...")
        self.line_edit.returnPressed.connect(self.add_custom_entry)
        self.list_widget = QListWidget()
        self.list_widget.setFlow(QListWidget.LeftToRight)
        self.list_widget.setWrapping(True)
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setSpacing(4)
        self.list_widget.setMaximumHeight(100)
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
        
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.currentRowChanged.connect(self.change_tab)
        
        self.stacked_widget = QStackedWidget()
        
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stacked_widget)
        
        self.setCentralWidget(main_widget)

        self.home_tab = QWidget()
        self.event_sheet_tab = QWidget()
        self.file_pair_tab = QWidget()
        self.options_tab = QWidget()
        self.analysis_tab = QWidget()
        self.results_tab = QWidget()
        self.visualization_tab = QWidget()

        self.add_sidebar_tab(self.home_tab, "Home")
        self.add_sidebar_tab(self.event_sheet_tab, "Event Sheet")
        self.add_sidebar_tab(self.file_pair_tab, "File Pair")
        self.add_sidebar_tab(self.options_tab, "Options")
        self.add_sidebar_tab(self.analysis_tab, "Analysis")
        self.add_sidebar_tab(self.results_tab, "Results")
        self.add_sidebar_tab(self.visualization_tab, "Visualization")
        
        self.init_status_bar()
        self.apply_theme()
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

    def add_sidebar_tab(self, widget, name):
        self.stacked_widget.addWidget(widget)
        item = QListWidgetItem(name)
        item.setSizeHint(QSize(0, 40))
        item.setFont(QFont("Inter", 12))
        self.sidebar.addItem(item)
        
    def change_tab(self, index):
        if self.sidebar.item(index).text() == "Analysis":
            event_sheet_path = self.event_file_path.text()
            file_pair_path = self.file_pair_path.text()
            if not os.path.exists(event_sheet_path) or not os.path.exists(file_pair_path):
                QMessageBox.warning(self, "Prerequisites Missing", "Please load an Event Sheet and File Pair sheet before proceeding to Analysis.")
                self.sidebar.setCurrentRow(1) # Reverts to Event Sheet tab
                return
        self.stacked_widget.setCurrentIndex(index)
        
    def init_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready")
        
    def apply_theme(self):
        qss = """
        QMainWindow { background-color: #1e1e2e; color: #cdd6f4; }
        QWidget {
            background-color: #1e1e2e;
            color: #cdd6f4;
            font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
            font-size: 13px;
        }
        QPushButton {
            background-color: #89b4fa;
            color: #11111b;
            border: none;
            border-radius: 4px;
            padding: 6px 16px;
            font-weight: bold;
        }
        QPushButton:hover { background-color: #b4befe; }
        QPushButton:pressed { background-color: #74c7ec; }
        QPushButton:disabled { background-color: #45475a; color: #6c7086; }
        QLineEdit, QComboBox, QSpinBox {
            background-color: #313244;
            border: 1px solid #45475a;
            border-radius: 4px;
            padding: 4px;
            color: #cdd6f4;
        }
        QTableWidget {
            background-color: #1e1e2e;
            alternate-background-color: #313244;
            gridline-color: #45475a;
            border: 1px solid #45475a;
            border-radius: 4px;
        }
        QHeaderView::section {
            background-color: #313244;
            color: #cdd6f4;
            padding: 4px;
            border: none;
            font-weight: bold;
        }
        QGroupBox {
            border: 1px solid #45475a;
            border-radius: 6px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 4px;
            color: #89b4fa;
        }
        QListWidget#Sidebar {
            background-color: #181825;
            border-right: 1px solid #313244;
            padding-top: 10px;
        }
        QListWidget#Sidebar::item {
            padding: 10px;
            margin: 4px 10px;
            border-radius: 6px;
        }
        QListWidget#Sidebar::item:hover { background-color: #313244; }
        QListWidget#Sidebar::item:selected {
            background-color: #89b4fa;
            color: #11111b;
            font-weight: bold;
        }
        QProgressBar {
            border: 1px solid #45475a;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #a6e3a1;
            width: 20px;
        }
        QStatusBar { background-color: #181825; color: #cdd6f4; }
        """
        self.setStyleSheet(qss)

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

    def apply_template_behaviour(self, file_path):
        """Parse the template behaviour file and update the UI with its contents."""
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
            
            # Downstream updates
            self.update_options_with_template()
            
            # If the event sheet is already loaded, we might want to refresh its dropdowns
            # but since this is usually called before or during loading, we check if it has data
            if hasattr(self, 'event_file_path') and self.event_file_path.text() and os.path.exists(self.event_file_path.text()):
                self.display_csv_in_table(self.event_file_path.text(), self.event_table)
                
            return True
        except Exception as e:
            raise e

    def import_template_behaviour_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Import Template Behaviour File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.apply_template_behaviour(file_path)
                QMessageBox.information(self, "Imported", f"Template behaviour file imported from: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load template: {str(e)}")

    def init_home_tab(self):
        layout = QVBoxLayout()
        
        # Welcome Header
        welcome_label = QLabel("Welcome to the Fiber Photometry Data Analyzer")
        welcome_label.setFont(QFont("Inter", 18, QFont.Bold))
        layout.addWidget(welcome_label)
        
        # System Info Box
        info_group = QGroupBox("System Status")
        info_layout = QFormLayout()
        import platform
        info_layout.addRow("Python Version:", QLabel(sys.version.split(' ')[0]))
        info_layout.addRow("OS Platform:", QLabel(f"{platform.system()} {platform.release()}"))
        try:
            cores = str(os.cpu_count())
        except Exception:
            cores = "Unknown"
        info_layout.addRow("CPU Cores Available:", QLabel(cores))
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Quick Start Guide
        guide_group = QGroupBox("Quick Start Guide")
        guide_layout = QVBoxLayout()
        guide_text = QLabel(
            "1. Go to 'Event Sheet' to load or create your event definitions.\n"
            "2. Go to 'File Pair' to link Doric and Abet files.\n"
            "3. Configure your processing parameters in the 'Options' tab.\n"
            "4. Run your analysis from the 'Analysis' tab.\n"
            "5. View results and generate plots in the 'Results' and 'Visualization' tabs."
        )
        guide_text.setWordWrap(True)
        guide_layout.addWidget(guide_text)
        guide_group.setLayout(guide_layout)
        layout.addWidget(guide_group)
        
        layout.addStretch()
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
        self.event_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.event_table.customContextMenuRequested.connect(lambda pos: self.show_table_context_menu(pos, self.event_table))
        layout.addWidget(self.event_table)
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save to CSV")
        save_btn.clicked.connect(lambda: self.save_table_to_csv(self.event_table))
        button_layout.addWidget(save_btn)
        layout.addStretch()
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
        self.file_pair_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_pair_table.customContextMenuRequested.connect(lambda pos: self.show_table_context_menu(pos, self.file_pair_table))
        layout.addWidget(self.file_pair_table)
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save to CSV")
        save_btn.clicked.connect(lambda: self.save_table_to_csv(self.file_pair_table))
        button_layout.addWidget(save_btn)
        layout.addStretch()
        layout.addLayout(button_layout)
        self.file_pair_tab.setLayout(layout)

    def show_table_context_menu(self, pos, table_widget):
        menu = QMenu()
        add_action = menu.addAction("Add Row")
        remove_action = menu.addAction("Remove Row")
        action = menu.exec(table_widget.viewport().mapToGlobal(pos))
        if action == add_action:
            self.add_row(table_widget)
        elif action == remove_action:
            self.remove_row(table_widget)

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
            # Pre-fetch values for this row to use if dropdowns are created
            init_vals = {}
            for c in range(cols):
                h = data.columns[c].lower()
                init_vals[h] = str(data.iat[row, c]) if pd.notna(data.iat[row, c]) else ""

            for col in range(cols):
                header = data.columns[col].lower()
                cell_val = init_vals.get(header, "")

                if header == 'event_type' and self.template_loaded:
                    event_type_combo = QComboBox()
                    event_type_combo.addItems(self.unique_event_types)
                    event_type_combo.setCurrentText(cell_val)
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
                    try:
                        f_val = int(float(cell_val)) if cell_val else 0
                    except ValueError:
                        f_val = 0
                    num_filter_spinbox.setValue(f_val)
                    num_filter_spinbox.valueChanged.connect(lambda value, r=row: self.adjust_filter_columns(r, value, table_widget))
                    table_widget.setCellWidget(row, col, num_filter_spinbox)
                elif header in ('abet_path', 'doric_path'):
                    fp_widget = FilePathWidget(cell_val)
                    table_widget.setCellWidget(row, col, fp_widget)
                else:
                    item = QTableWidgetItem(cell_val)
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    table_widget.setItem(row, col, item)
            
            # After widgets are created for the row, initialize the dropdown cascade if applicable
            if self.template_loaded and 'event_type' in init_vals:
                self.filter_event_data(row, init_vals['event_type'], table_widget, 
                                       initial_name=init_vals.get('event_name'), 
                                       initial_group=init_vals.get('event_group'))
        # Ensure columns and rows are sized to show content and refresh the widget so rows become visible
        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()
        try:
            table_widget.scrollToTop()
        except Exception:
            pass
        table_widget.repaint()
        table_widget.setVisible(True)

    def filter_event_data(self, row, event_type, table_widget, initial_name=None, initial_group=None):
        if not self.template_loaded:
            return
        filtered_names = self.event_data[self.event_data['Evnt_Name'] == event_type]['Item_Name'].dropna().unique()
        name_combo = table_widget.cellWidget(row, 1)
        if not isinstance(name_combo, QComboBox):
            return
            
        name_combo.blockSignals(True)
        name_combo.clear()
        name_combo.addItems(filtered_names)
        if initial_name and initial_name in filtered_names:
            name_combo.setCurrentText(initial_name)
        name_combo.blockSignals(False)
        
        current_name = name_combo.currentText()
        self.filter_event_group(row, current_name, table_widget, initial_group=initial_group)

    def filter_event_group(self, row, event_name, table_widget, initial_group=None):
        if not self.template_loaded:
            return
        event_type_combo = table_widget.cellWidget(row, 0)
        if not isinstance(event_type_combo, QComboBox):
            return
        event_type = event_type_combo.currentText()
        filtered_groups = self.event_data[(self.event_data['Evnt_Name'] == event_type) & (self.event_data['Item_Name'] == event_name)]['Group_ID'].dropna().unique()
        group_combo = table_widget.cellWidget(row, 2)
        if not isinstance(group_combo, QComboBox):
            return
            
        group_combo.blockSignals(True)
        group_combo.clear()
        group_combo.addItems(filtered_groups)
        if initial_group and initial_group in filtered_groups:
            group_combo.setCurrentText(initial_group)
        group_combo.blockSignals(False)

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

        # Initialize dropdowns for the new row if template is loaded
        if self.template_loaded:
            type_combo = table_widget.cellWidget(current_row_count, 0)
            if isinstance(type_combo, QComboBox) and type_combo.count() > 0:
                self.filter_event_data(current_row_count, type_combo.currentText(), table_widget)

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
        self.progress_bar.show()
        self.status_bar.showMessage("Running Analysis...")
        self._analysis_thread.start()

    def _on_analysis_complete(self, results, combined_results):
        """Called on the main thread when the analysis worker finishes."""
        self.analysis_results = results
        self.combined_results = combined_results
        self.run_analysis_button.setEnabled(True)
        self.run_analysis_button.setText("Run Analysis")
        self.progress_bar.hide()
        self.status_bar.showMessage("Analysis Complete")
        self.update_results_and_visualization_options()
        QMessageBox.information(self, "Analysis", "Analysis complete!")

    def _on_analysis_error(self, error_msg):
        """Called on the main thread when the analysis worker raises an exception."""
        self.run_analysis_button.setEnabled(True)
        self.run_analysis_button.setText("Run Analysis")
        self.progress_bar.hide()
        self.status_bar.showMessage("Analysis Error")
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
        self.results_date_select.currentTextChanged.connect(self.update_results_behavior_select)
        self.results_behavior_select = QComboBox()
        self.results_behavior_select.currentTextChanged.connect(self.update_results_table)
        controls_layout.addRow("Select Animal ID:", self.results_animal_select)
        controls_layout.addRow("Select Date:", self.results_date_select)
        controls_layout.addRow("Select Behavior:", self.results_behavior_select)
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
        self.vis_animal_select = MultiSelectComboBox()
        self.vis_animal_select.list_widget.itemChanged.connect(self.update_vis_date_select)
        
        self.vis_animal_treatment = QComboBox()
        self.vis_animal_treatment.addItems(["Combine", "Separate Lines", "Separate Subplots"])
        
        self.vis_date_select = MultiSelectComboBox()
        self.vis_date_select.list_widget.itemChanged.connect(self.update_vis_behavior_select)
        
        self.vis_date_mode = QComboBox()
        self.vis_date_mode.addItems(["Date", "Date Time", "Order"])
        self.vis_date_mode.currentTextChanged.connect(self.update_vis_date_select)
        
        self.vis_date_treatment = QComboBox()
        self.vis_date_treatment.addItems(["Combine", "Separate Lines", "Separate Subplots"])
        
        self.vis_behavior_select = MultiSelectComboBox()
        
        self.vis_behavior_treatment = QComboBox()
        self.vis_behavior_treatment.addItems(["Combine", "Separate Lines", "Separate Subplots"])
        
        # populate selects from any existing results
        self.update_animal_date_selects()
        
        self.event_prior_input = QLineEdit(self.config['Event_Window'].get('event_prior', '5.0'))
        self.event_prior_input.setDisabled(True)
        self.event_follow_input = QLineEdit(self.config['Event_Window'].get('event_follow', '10.0'))
        self.event_follow_input.setDisabled(True)
        
        # Visualization mode selector
        self.vis_mode_select = QComboBox()
        self.vis_mode_select.addItems(["Histogram", "Heatmap"])
        
        # Color Scheme selector
        self.vis_color_scheme = QComboBox()
        self.vis_color_scheme.addItems(["Default", "Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Coolwarm"])
        
        animal_layout = QHBoxLayout()
        animal_layout.addWidget(self.vis_animal_select, 2)
        animal_layout.addWidget(self.vis_animal_treatment, 1)
        controls_layout.addRow("Select Animal ID:", animal_layout)
        
        date_layout = QHBoxLayout()
        date_layout.addWidget(self.vis_date_mode, 1)
        date_layout.addWidget(self.vis_date_select, 2)
        date_layout.addWidget(self.vis_date_treatment, 1)
        controls_layout.addRow("Select Date:", date_layout)
        
        behavior_layout = QHBoxLayout()
        behavior_layout.addWidget(self.vis_behavior_select, 2)
        behavior_layout.addWidget(self.vis_behavior_treatment, 1)
        controls_layout.addRow("Select Behavior:", behavior_layout)
        
        display_layout = QHBoxLayout()
        display_layout.addWidget(self.vis_mode_select, 1)
        display_layout.addWidget(QLabel("Color Scheme:"), 0)
        display_layout.addWidget(self.vis_color_scheme, 1)
        controls_layout.addRow("Display Options:", display_layout)
        
        # Axis Limits
        axes_layout = QHBoxLayout()
        axes_layout.addWidget(QLabel("X Min:"))
        self.vis_x_min = QLineEdit()
        axes_layout.addWidget(self.vis_x_min)
        axes_layout.addWidget(QLabel("X Max:"))
        self.vis_x_max = QLineEdit()
        axes_layout.addWidget(self.vis_x_max)
        
        axes_layout.addWidget(QLabel("Y Min:"))
        self.vis_y_min = QLineEdit()
        axes_layout.addWidget(self.vis_y_min)
        axes_layout.addWidget(QLabel("Y Max:"))
        self.vis_y_max = QLineEdit()
        axes_layout.addWidget(self.vis_y_max)
        controls_layout.addRow("Manual Axis Limits:", axes_layout)
        
        generate_plot_button = QPushButton("Generate Plot")
        generate_plot_button.clicked.connect(self.generate_plot)
        
        layout.addLayout(controls_layout)
        layout.addWidget(generate_plot_button)
        
        self.canvas = MatplotlibCanvas(self, width=5, height=4, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
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
            self.vis_animal_select.list_widget.clear()
            self.vis_animal_select.add_option('All')
            
            if combined_exists:
                self.vis_animal_select.add_option('Combined')
            for a in animals:
                self.vis_animal_select.add_option(str(a))
                
            self.vis_date_select.list_widget.clear()
            self.update_vis_date_select()

    def update_vis_date_select(self, *_):
        """Populate the visualization date select when the animal selection changes."""
        self.vis_date_select.list_widget.clear()
        selected_animals = self.vis_animal_select.get_checked_items() if hasattr(self, 'vis_animal_select') else []
        if not selected_animals or not self.analysis_results:
            return
        
        mode = self.vis_date_mode.currentText() if hasattr(self, 'vis_date_mode') else "Date"
        dates = set()
        for res in self.analysis_results:
            a = res.get('animal_id')
            if a is None: continue
            if 'All' in selected_animals or str(a) in selected_animals:
                if mode == "Date":
                    d = res.get('date')
                elif mode == "Date Time":
                    d = res.get('datetime', res.get('date'))
                else:
                    d = res.get('session', res.get('date'))
                if d is not None:
                    dates.add(d)

        # allow viewing all dates for the animal by adding options
        self.vis_date_select.add_option('All')
        try:
            sorted_dates = sorted(dates)
        except TypeError:
            sorted_dates = sorted(dates, key=str)
            
        for d in sorted_dates:
            self.vis_date_select.add_option(str(d))
        self.update_vis_behavior_select()

    def update_results_date_select(self, animal: str):
        """Populate the results date select when the results animal selection changes."""
        self.results_date_select.clear()
        if not animal or not self.analysis_results:
            return
        if animal == 'Combined':
            self.results_date_select.addItem('Combined')
            self.update_results_behavior_select()
            return
        dates = sorted({res.get('date') for res in self.analysis_results if res.get('animal_id') == animal and res.get('date')})
        self.results_date_select.addItem('All')
        for d in dates:
            self.results_date_select.addItem(str(d))
        self.update_results_behavior_select()

    def update_results_behavior_select(self, *_):
        """Populate the results behavior select based on animal and date selections."""
        self.results_behavior_select.clear()
        selected_animal = self.results_animal_select.currentText() if hasattr(self, 'results_animal_select') else ''
        selected_date = self.results_date_select.currentText() if hasattr(self, 'results_date_select') else ''
        
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
            
        behaviors = sorted(list({res.get('behavior') for res in self.analysis_results if matches(res) and res.get('behavior')}))
        self.results_behavior_select.addItem('All')
        for b in behaviors:
            self.results_behavior_select.addItem(str(b))
        self.update_results_table()

    def update_results_table(self):
        selected_animal = self.results_animal_select.currentText() if hasattr(self, 'results_animal_select') else ''
        selected_date = self.results_date_select.currentText() if hasattr(self, 'results_date_select') else ''
        selected_behavior = self.results_behavior_select.currentText() if hasattr(self, 'results_behavior_select') else ''
        
        if not selected_animal or not self.analysis_results:
            self.results_table.clear()
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            return

        # Filter results for the selected animal, date, and behavior
        def matches_selection(res):
            res_animal = res.get('animal_id')
            res_date = res.get('date')
            res_behavior = res.get('behavior')
            
            if selected_animal == 'Combined':
                if res_animal:
                    return False
            else:
                if res_animal != selected_animal:
                    return False
            if selected_date and selected_date not in ['Combined', 'All']:
                if res_date != selected_date:
                    return False
            if selected_behavior and selected_behavior != 'All':
                if res_behavior != selected_behavior:
                    return False
            return True

        filtered_results = [res for res in self.analysis_results if matches_selection(res)]

        # Determine dynamic columns
        include_animal = (selected_animal == 'Combined')
        include_date = (selected_date in ['Combined', 'All'])
        
        columns = []
        if include_animal:
            columns.append('Animal')
        if include_date:
            columns.append('Date')
            columns.append('Time')
        columns.extend(['Behavior', 'Max Peak', 'AUC'])

        self.results_table.setRowCount(len(filtered_results))
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        
        for i, res in enumerate(filtered_results):
            col_idx = 0
            if include_animal:
                self.results_table.setItem(i, col_idx, QTableWidgetItem(str(res.get('animal_id', 'Combined'))))
                col_idx += 1
            if include_date:
                self.results_table.setItem(i, col_idx, QTableWidgetItem(str(res.get('date', 'All'))))
                col_idx += 1
                self.results_table.setItem(i, col_idx, QTableWidgetItem(str(res.get('time', ''))))
                col_idx += 1
                
            self.results_table.setItem(i, col_idx, QTableWidgetItem(str(res.get('behavior', ''))))
            col_idx += 1
            max_peak = res.get('max_peak')
            auc = res.get('auc')
            self.results_table.setItem(i, col_idx, QTableWidgetItem(f"{float(max_peak):.4f}" if max_peak is not None else 'N/A'))
            col_idx += 1
            self.results_table.setItem(i, col_idx, QTableWidgetItem(f"{float(auc):.4f}" if auc is not None else 'N/A'))
            
        self.results_table.resizeColumnsToContents()

    def update_vis_file_select(self):
        # legacy helper left for compatibility; delegate to animal/date selector updater
        self.update_animal_date_selects()

    def update_vis_behavior_select(self, *_):
        self.vis_behavior_select.list_widget.clear()
        selected_animals = self.vis_animal_select.get_checked_items() if hasattr(self, 'vis_animal_select') else []
        selected_dates = self.vis_date_select.get_checked_items() if hasattr(self, 'vis_date_select') else []
        
        if not selected_animals or not selected_dates or not self.analysis_results:
            return

        mode = self.vis_date_mode.currentText() if hasattr(self, 'vis_date_mode') else "Date"

        def matches(res):
            a = str(res.get('animal_id'))
            if mode == "Date":
                d = str(res.get('date'))
            elif mode == "Date Time":
                d = str(res.get('datetime', res.get('date')))
            else:
                d = str(res.get('session', res.get('date')))
                
            if 'All' not in selected_animals and a not in selected_animals:
                return False
            if 'All' not in selected_dates and d not in selected_dates:
                return False
            return True

        behaviors = sorted(list({res.get('behavior') for res in self.analysis_results if matches(res) and res.get('behavior')}))
        self.vis_behavior_select.add_option('All')
        for b in behaviors:
            self.vis_behavior_select.add_option(str(b))

    def generate_plot(self):
        # Extract selections from MultiSelectComboBox
        selected_animals = self.vis_animal_select.get_checked_items() if hasattr(self, 'vis_animal_select') else []
        selected_dates = self.vis_date_select.get_checked_items() if hasattr(self, 'vis_date_select') else []
        selected_behaviors = self.vis_behavior_select.get_checked_items() if hasattr(self, 'vis_behavior_select') else []
        
        if not selected_behaviors:
            QMessageBox.warning(self, "Selection Error", "Please select at least one behavior to plot.")
            return
            
        # Expand 'All' selections to represent all unique values
        if 'All' in selected_animals:
            selected_animals = [str(self.vis_animal_select.list_widget.item(i).text()) 
                                for i in range(self.vis_animal_select.list_widget.count()) 
                                if str(self.vis_animal_select.list_widget.item(i).text()) not in ['All', 'Combined']]
                                
        if 'All' in selected_dates:
            selected_dates = [str(self.vis_date_select.list_widget.item(i).text()) 
                              for i in range(self.vis_date_select.list_widget.count()) 
                              if str(self.vis_date_select.list_widget.item(i).text()) not in ['All', 'Combined']]
                              
        if 'All' in selected_behaviors:
            selected_behaviors = [str(self.vis_behavior_select.list_widget.item(i).text()) 
                                  for i in range(self.vis_behavior_select.list_widget.count()) 
                                  if str(self.vis_behavior_select.list_widget.item(i).text()) not in ['All', 'Combined']]

        try:
            event_prior = float(self.event_prior_input.text() or 5.0)
            event_follow = float(self.event_follow_input.text() or 10.0)
        except ValueError:
            event_prior, event_follow = 5.0, 10.0

        vis_mode = self.vis_mode_select.currentText().lower() if hasattr(self, 'vis_mode_select') else 'histogram'
        color_scheme = self.vis_color_scheme.currentText().lower() if hasattr(self, 'vis_color_scheme') else 'default'
        
        if color_scheme == "default":
            cmap = plt.get_cmap("tab10")
        else:
            try:
                cmap = plt.get_cmap(color_scheme)
            except ValueError:
                cmap = plt.get_cmap("tab10")
                
        animal_treatment = self.vis_animal_treatment.currentText()
        date_treatment = self.vis_date_treatment.currentText()
        behavior_treatment = self.vis_behavior_treatment.currentText()
        
        mode = self.vis_date_mode.currentText() if hasattr(self, 'vis_date_mode') else "Date"

        grouped_raw_results = {}
        for res in self.analysis_results:
            a = str(res.get('animal_id'))
            b = str(res.get('behavior', ''))
            
            if mode == "Date":
                d = str(res.get('date'))
            elif mode == "Date Time":
                d = str(res.get('datetime', res.get('date')))
            else:
                d = str(res.get('session', res.get('date')))
                
            # Filter against UI selections
            if a not in selected_animals:
                continue
            if d not in selected_dates:
                continue
            if b not in selected_behaviors:
                continue
                
            if res.get('plot_data').empty:
                continue
                
            # Apply combination treatments immediately internally
            eff_a = 'Combined' if animal_treatment == "Combine" else a
            eff_d = 'All' if date_treatment == "Combine" else d
            eff_b = 'Combined' if behavior_treatment == "Combine" else b
            
            key = (eff_a, eff_d, eff_b)
            if key not in grouped_raw_results:
                grouped_raw_results[key] = []
            grouped_raw_results[key].append(res['plot_data'])

        matched_results = []
        for (eff_a, eff_d, eff_b), data_list in grouped_raw_results.items():
            if len(data_list) == 1:
                combined_plot_data = data_list[0]
            else:
                try:
                    combined_plot_data = pd.concat(data_list, axis=1)
                except Exception:
                    combined_plot_data = data_list[0]
            
            matched_results.append({
                'animal': eff_a,
                'date': eff_d,
                'behavior': eff_b,
                'data': combined_plot_data
            })

        if not matched_results:
            self.canvas.axes.clear()
            self.canvas.draw()
            QMessageBox.warning(self, "Data Not Found", f"No data matches the selected criteria or the data is empty.")
            return

        # Store first plot data so save functionality works as expected for simple cases
        self.plot_data = matched_results[0]['data']

        self.canvas.figure.clf()
        
        # Check treatments
        animal_treatment = self.vis_animal_treatment.currentText()
        date_treatment = self.vis_date_treatment.currentText()
        behavior_treatment = self.vis_behavior_treatment.currentText()
        
        # Determine number of subplots
        subplots_needed = 1
        subplot_groups = {} # group results by their subplot index
        group_keys = []
        
        for res in matched_results:
            # Create a tuple representing which subplot this result belongs to
            sp_key = []
            if animal_treatment == "Separate Subplots": sp_key.append(res['animal'])
            if date_treatment == "Separate Subplots": sp_key.append(res['date'])
            if behavior_treatment == "Separate Subplots": sp_key.append(res['behavior'])
            
            sp_key = tuple(sp_key)
            if sp_key not in subplot_groups:
                subplot_groups[sp_key] = []
                group_keys.append(sp_key)
            subplot_groups[sp_key].append(res)
            
        subplots_needed = max(1, len(subplot_groups))
        
        # Create columns/rows for subplots
        cols = 1
        if subplots_needed > 1:
            cols = 2 if subplots_needed % 2 == 0 or subplots_needed > 3 else 1
        rows = (subplots_needed + cols - 1) // cols
        
        for idx, (sp_key, Group) in enumerate(subplot_groups.items()):
            ax = self.canvas.figure.add_subplot(rows, cols, idx + 1)
            if idx == 0:
                self.canvas.axes = ax # Set primary axes
                
            if vis_mode == 'histogram':
                ax.axvline(x=0, color='r', linestyle='--')
                for j, res in enumerate(Group):
                    data = res['data']
                    label_parts = []
                    if animal_treatment != "Combine" and res['animal'] != 'All': label_parts.append(res['animal'])
                    if date_treatment != "Combine" and res['date'] != 'All': label_parts.append(res['date'])
                    if behavior_treatment != "Combine": label_parts.append(res['behavior'])
                    label = " | ".join(label_parts) if label_parts else "Mean"
                    
                    time_axis = np.linspace(-event_prior, event_follow, len(data.index))
                    c_idx = (idx * 5 + j) % cmap.N
                    if res['animal'] == 'Combined' or res['date'] == 'All':
                        mean_data = data.mean(axis=1)
                        sem_data = data.sem(axis=1)
                        ax.plot(time_axis, mean_data, label=label, linewidth=2, color=cmap(c_idx))
                        ax.fill_between(time_axis, mean_data - sem_data, mean_data + sem_data, alpha=0.2, color=cmap(c_idx))
                    elif ' (Δ' in res['behavior']:
                        for col in data.columns:
                            ax.plot(time_axis, data[col], linewidth=2, label=str(col))
                    else:
                        mean_data = data.mean(axis=1)
                        sem_data = data.sem(axis=1)
                        ax.plot(time_axis, mean_data, label=label, color=cmap(c_idx))
                        ax.fill_between(time_axis, mean_data - sem_data, mean_data + sem_data, alpha=0.2, color=cmap(c_idx))
                
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Signal")
                title_parts = [str(k) for k in sp_key] if sp_key else ["Perievent Histogram"]
                ax.set_title(" - ".join(title_parts))
                if len(ax.get_legend_handles_labels()[0]) > 0:
                    if len(ax.get_legend_handles_labels()[0]) > 6:
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
                    else:
                        ax.legend()
                        
            elif vis_mode == 'heatmap':
                # Just plot the first one in the group forheatmap since combining them on 1 heatmap is harder
                res = Group[0]
                data = res['data'].T.values 
                im = ax.imshow(data, aspect='auto', cmap=color_scheme.lower() if color_scheme != 'default' else 'viridis',
                               extent=[-event_prior, event_follow, data.shape[0], 0])
                ax.axvline(x=0, color='r', linestyle='--')
                ax.set_xlabel("Time (s)")
                
                if res['animal'] == 'Combined':
                    ax.set_ylabel("Subject/Animal")
                    if data.shape[0] <= 15:
                        ax.set_yticks(np.arange(data.shape[0]) + 0.5)
                        ax.set_yticklabels(res['data'].columns)
                elif ' (Δ' in res['behavior']:
                    ax.set_ylabel("Difference Trace")
                    ax.set_yticks([0.5])
                    ax.set_yticklabels([res['data'].columns[0]])
                else:
                    ax.set_ylabel("Event Instance (Trial)")
                    
                title_parts = [str(k) for k in sp_key] if sp_key else [res['behavior']]
                ax.set_title(" Heatmap - ".join(title_parts))
                
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = self.canvas.figure.colorbar(im, cax=cax)
                cbar.set_label("Signal")
                
        # Apply Manual Axis Limits
        if hasattr(self, 'vis_y_min') and self.vis_y_min.text() and vis_mode != 'heatmap':
            try: ax.set_ylim(bottom=float(self.vis_y_min.text()))
            except ValueError: pass
        if hasattr(self, 'vis_y_max') and self.vis_y_max.text() and vis_mode != 'heatmap':
            try: ax.set_ylim(top=float(self.vis_y_max.text()))
            except ValueError: pass
        if hasattr(self, 'vis_x_min') and self.vis_x_min.text():
            try: ax.set_xlim(left=float(self.vis_x_min.text()))
            except ValueError: pass
        if hasattr(self, 'vis_x_max') and self.vis_x_max.text():
            try: ax.set_xlim(right=float(self.vis_x_max.text()))
            except ValueError: pass

        self.canvas.figure.tight_layout()
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
        file_path_keys = {'file_list_path': 'open', 'event_list_path': 'open', 'output_path': 'directory', 'template_path': 'open'}
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
            
            # Add preview button underneath Photometry Processing group
            if section == "Photometry_Processing":
                preview_btn = QPushButton("Preview Photometry Signal")
                preview_btn.clicked.connect(self.open_photometry_preview)
                content_layout.addWidget(preview_btn)
                
        save_button = QPushButton("Save Changes")
        save_button.clicked.connect(self.save_config_changes_to_current_file)
        content_layout.addWidget(save_button)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)
        outer_layout.addWidget(scroll)
        self.options_tab.setLayout(outer_layout)

    def open_photometry_preview(self):
        # Read the file pair table to get the paths
        file_sheet_path = self.file_pair_path.text()
        file_pair_data = None
        if os.path.exists(file_sheet_path):
            try:
                file_pair_data = pd.read_csv(file_sheet_path)
            except Exception as e:
                print(f"Failed to read file pair sheet for preview: {e}")
                
        preview_dialog = PhotometryPreviewWindow(self.config, file_pair_data, parent=self)
        preview_dialog.exec()

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
          - template_path   -> template behaviour (apply_template_behaviour)
        This will set the QLineEdit paths and call display_csv_in_table to populate the tables.
        """
        try:
            if "Filepath" not in self.config:
                return

            # 1. Load template file FIRST so that subsequent table loading can use dropdowns
            template_path = self.config['Filepath'].get('template_path', '').strip()
            if template_path:
                if os.path.isfile(template_path):
                    try:
                        self.apply_template_behaviour(template_path)
                        print(f"Loaded template behaviour file from config: {template_path}")
                    except Exception as e:
                        print(f"Warning: Failed to auto-load template sheet from config: {e}")
                else:
                    print(f"Warning: Template path from config is not a valid file: {template_path}")

            file_list_path = self.config['Filepath'].get('file_list_path', '').strip()
            event_list_path = self.config['Filepath'].get('event_list_path', '').strip()

            # 2. Load event sheet if provided and is a file
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

            # 3. Load file pair sheet if provided and is a file
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