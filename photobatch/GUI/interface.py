import sys
import pandas as pd
import os
import pickle
import shutil
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
from photobatch.Processing import data_processor, hdf_store

# type: ignore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QDialog

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.tight_layout()
        super(MatplotlibCanvas, self).__init__(fig)


class PlotViewerWindow(QDialog):
    def __init__(self, source_figure, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Viewer")
        self.resize(1600, 900)

        layout = QVBoxLayout(self)
        cloned_figure = pickle.loads(pickle.dumps(source_figure))
        self.canvas = FigureCanvas(cloned_figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.showMaximized()

class PhotometryPreviewWindow(QDialog):
    def __init__(self, config, file_pair_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Photometry Preview Window")
        self.resize(2000, 900)
        self.config = config
        self.file_pair_data = file_pair_data
        
        main_layout = QHBoxLayout(self)
        
        # Left side controls (scrollable so they never get clipped)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFixedWidth(320)
        controls_inner = QWidget()
        controls_layout = QVBoxLayout(controls_inner)
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
        
        self._fo_cutoff_label = QLabel("Cutoff/Savgol Window:")
        self.filter_cutoff = QSpinBox()
        self.filter_cutoff.setRange(1, 1000)
        self.filter_cutoff.setValue(int(self.config['Photometry_Processing'].get('filter_cutoff', '10')))
        controls_form.addRow(self._fo_cutoff_label, self.filter_cutoff)
        
        # cheby_ripple: only shown when filter_type==lowpass AND filter_name==chebychev
        self._cheby_ripple_label = QLabel("Chebyshev Ripple (dB):")
        self.cheby_ripple = QLineEdit(self.config['Photometry_Processing'].get('cheby_ripple', '1.0'))
        controls_form.addRow(self._cheby_ripple_label, self.cheby_ripple)
        
        self.crop_start = QLineEdit(self.config['Photometry_Processing'].get('crop_start', '0.0'))
        controls_form.addRow("Crop Start (s):", self.crop_start)
        
        self.crop_end = QLineEdit(self.config['Photometry_Processing'].get('crop_end', '0.0'))
        controls_form.addRow("Crop End (s):", self.crop_end)
        
        # Despike group
        self.despike = QCheckBox()
        despike_str = self.config['Photometry_Processing'].get('despike', 'true')
        self.despike.setChecked(despike_str.lower() in ('true', '1'))
        controls_form.addRow("Despike:", self.despike)
        
        self._despike_window_label = QLabel("Despike Window:")
        self.despike_window = QSpinBox()
        self.despike_window.setRange(3, 100001)
        self.despike_window.setSingleStep(2)
        try:
            self.despike_window.setValue(int(self.config['Photometry_Processing'].get('despike_window', '2001')))
        except ValueError:
            self.despike_window.setValue(2001)
        controls_form.addRow(self._despike_window_label, self.despike_window)
        
        self._despike_threshold_label = QLabel("Despike Threshold:")
        self.despike_threshold = QLineEdit(self.config['Photometry_Processing'].get('despike_threshold', '5.0'))
        controls_form.addRow(self._despike_threshold_label, self.despike_threshold)
        
        self.fit_type = QComboBox()
        self.fit_type.addItems(['linear', 'expodecay', 'arpls'])
        self.fit_type.setCurrentText(self.config['Photometry_Processing'].get('fit_type', 'linear'))
        controls_form.addRow("Fit Type:", self.fit_type)
        
        # robust_fit: only shown when fit_type == linear
        self._robust_fit_label = QLabel("Robust Fit (Huber):")
        self.robust_fit = QCheckBox()
        robust_str = self.config['Photometry_Processing'].get('robust_fit', 'true')
        self.robust_fit.setChecked(robust_str.lower() in ('true', '1'))
        controls_form.addRow(self._robust_fit_label, self.robust_fit)
        
        # huber_epsilon: only shown when fit_type == linear AND robust_fit is checked
        self._huber_epsilon_label = QLabel("Huber Epsilon:")
        self.huber_epsilon = QLineEdit(self.config['Photometry_Processing'].get('huber_epsilon', 'auto'))
        self.huber_epsilon.setToolTip("'auto' or 'mad' = calculate from MAD of noise floor; or enter a numeric value (must be > 1.0)")
        controls_form.addRow(self._huber_epsilon_label, self.huber_epsilon)
        
        # arPLS controls — only shown when fit_type == arpls
        self._arpls_lambda_label = QLabel("arPLS Lambda:")
        self.arpls_lambda = QLineEdit(self.config['Photometry_Processing'].get('arpls_lambda', '1e5'))
        controls_form.addRow(self._arpls_lambda_label, self.arpls_lambda)
        
        self._arpls_max_iter_label = QLabel("arPLS Max Iterations:")
        self.arpls_max_iter = QSpinBox()
        self.arpls_max_iter.setRange(1, 1000)
        try:
            self.arpls_max_iter.setValue(int(self.config['Photometry_Processing'].get('arpls_max_iter', '50')))
        except ValueError:
            self.arpls_max_iter.setValue(50)
        controls_form.addRow(self._arpls_max_iter_label, self.arpls_max_iter)
        
        self._arpls_tol_label = QLabel("arPLS Tolerance:")
        self.arpls_tol = QLineEdit(self.config['Photometry_Processing'].get('arpls_tol', '1e-6'))
        controls_form.addRow(self._arpls_tol_label, self.arpls_tol)
        
        self._arpls_eps_label = QLabel("arPLS Epsilon:")
        self.arpls_eps = QLineEdit(self.config['Photometry_Processing'].get('arpls_eps', '1e-8'))
        controls_form.addRow(self._arpls_eps_label, self.arpls_eps)
        
        self._arpls_weight_scale_label = QLabel("arPLS Weight Scale:")
        self.arpls_weight_scale = QLineEdit(self.config['Photometry_Processing'].get('arpls_weight_scale', '2.0'))
        controls_form.addRow(self._arpls_weight_scale_label, self.arpls_weight_scale)
        
        self._arpls_downsample_label = QLabel("arPLS Downsample:")
        self.arpls_downsample = QCheckBox()
        ds_str = self.config['Photometry_Processing'].get('arpls_downsample', 'true')
        self.arpls_downsample.setChecked(ds_str.lower() in ('true', '1'))
        controls_form.addRow(self._arpls_downsample_label, self.arpls_downsample)
        
        controls_layout.addLayout(controls_form)
        
        self.update_btn = QPushButton("Update Preview")
        self.update_btn.clicked.connect(self.update_preview)
        controls_layout.addWidget(self.update_btn)
        controls_layout.addStretch()
        controls_scroll.setWidget(controls_inner)
        
        # Connect all conditional visibility
        self.fit_type.currentTextChanged.connect(self._update_preview_visibility)
        self.filter_type.currentTextChanged.connect(self._update_preview_visibility)
        self.filter_name.currentTextChanged.connect(self._update_preview_visibility)
        self.despike.stateChanged.connect(self._update_preview_visibility)
        self.robust_fit.stateChanged.connect(self._update_preview_visibility)
        self._update_preview_visibility()
        
        # Right side plots
        plots_layout = QVBoxLayout()
        self.raw_canvas = MatplotlibCanvas(self, width=12, height=3)
        self.filtered_canvas = MatplotlibCanvas(self, width=12, height=3)
        self.fitted_canvas = MatplotlibCanvas(self, width=12, height=3)
        
        plots_layout.addWidget(QLabel("Raw Data (Isobestic and Active)"))
        plots_layout.addWidget(self.raw_canvas)
        plots_layout.addWidget(QLabel("Filtered Data"))
        plots_layout.addWidget(self.filtered_canvas)
        plots_layout.addWidget(QLabel("Filtered & Fitted Data (Delta F/F)"))
        plots_layout.addWidget(self.fitted_canvas)
        
        main_layout.addWidget(controls_scroll)
        main_layout.addLayout(plots_layout, 1)

    def _update_preview_visibility(self):
        """Show/hide controls in the preview window based on current selections."""
        fit = self.fit_type.currentText()
        ftype = self.filter_type.currentText()
        fname = self.filter_name.currentText()
        is_arpls = fit == 'arpls'
        is_linear = fit == 'linear'
        is_cheby = ftype == 'lowpass' and fname == 'chebychev'
        is_despiking = self.despike.isChecked()

        for lbl, w in [
            (self._arpls_lambda_label, self.arpls_lambda),
            (self._arpls_max_iter_label, self.arpls_max_iter),
            (self._arpls_tol_label, self.arpls_tol),
            (self._arpls_eps_label, self.arpls_eps),
            (self._arpls_weight_scale_label, self.arpls_weight_scale),
            (self._arpls_downsample_label, self.arpls_downsample),
        ]:
            lbl.setVisible(is_arpls)
            w.setVisible(is_arpls)

        self._robust_fit_label.setVisible(is_linear)
        self.robust_fit.setVisible(is_linear)

        # Huber epsilon: visible only when fit_type == linear AND robust_fit is checked
        is_huber = is_linear and self.robust_fit.isChecked()
        self._huber_epsilon_label.setVisible(is_huber)
        self.huber_epsilon.setVisible(is_huber)

        self._cheby_ripple_label.setVisible(is_cheby)
        self.cheby_ripple.setVisible(is_cheby)

        self._despike_window_label.setVisible(is_despiking)
        self.despike_window.setVisible(is_despiking)
        self._despike_threshold_label.setVisible(is_despiking)
        self.despike_threshold.setVisible(is_despiking)
        
        
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
            
            # Parse despike params
            despike_enabled = self.despike.isChecked()
            despike_window_val = self.despike_window.value()
            try:
                despike_threshold_val = float(self.despike_threshold.text())
            except ValueError:
                despike_threshold_val = 5.0
            try:
                cheby_ripple_val = float(self.cheby_ripple.text())
            except ValueError:
                cheby_ripple_val = 1.0

            # Filter
            time_data, filtered_f0, filtered_f = photometry_data.doric_filter(
                filter_type=self.filter_type.currentText(),
                filter_name=self.filter_name.currentText(),
                filter_order=self.filter_order.value(),
                filter_cutoff=self.filter_cutoff.value(),
                despike=despike_enabled,
                despike_window=despike_window_val,
                despike_threshold=despike_threshold_val,
                cheby_ripple=cheby_ripple_val,
            )
            
            # Plot Filtered
            self.filtered_canvas.axes.clear()
            self.filtered_canvas.axes.plot(time_data, filtered_f0, label='Filtered Control', alpha=0.8)
            self.filtered_canvas.axes.plot(time_data, filtered_f, label='Filtered Active', alpha=0.8)
            self.filtered_canvas.axes.legend()
            self.filtered_canvas.axes.set_xlabel('Time (s)')
            self.filtered_canvas.axes.set_ylabel('Fluorescence')
            self.filtered_canvas.draw()
            
            # Parse fit params
            try:
                arpls_lambda_val = float(self.arpls_lambda.text())
            except ValueError:
                arpls_lambda_val = 1e5
            arpls_max_iter_val = self.arpls_max_iter.value()
            try:
                arpls_tol_val = float(self.arpls_tol.text())
            except ValueError:
                arpls_tol_val = 1e-6
            try:
                arpls_eps_val = float(self.arpls_eps.text())
            except ValueError:
                arpls_eps_val = 1e-8
            try:
                arpls_weight_scale_val = float(self.arpls_weight_scale.text())
            except ValueError:
                arpls_weight_scale_val = 2.0
            robust_fit_val = self.robust_fit.isChecked()
            huber_epsilon_val = self.huber_epsilon.text().strip()

            # Fit
            photometry_data.doric_fit(
                self.fit_type.currentText(), filtered_f0, filtered_f, time_data,
                robust_fit=robust_fit_val,
                huber_epsilon=huber_epsilon_val,
                arpls_lambda=arpls_lambda_val,
                arpls_max_iter=arpls_max_iter_val,
                arpls_tol=arpls_tol_val,
                arpls_eps=arpls_eps_val,
                arpls_weight_scale=arpls_weight_scale_val,
            )
            
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
    results_ready = Signal(str)
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
            hdf5_path = data_processor.process_files(
                self.file_pair_path,
                self.event_sheet_path,
                self.output_selections,
                self.config,
                self.num_workers
            )
            self.results_ready.emit(hdf5_path)
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

        self.results_hdf5_path = ''
        self.plot_data = pd.DataFrame()
        self.plot_window = None

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
        file_menu.addSeparator()
        open_analysis_action = QAction("Open Analysis Data...", self)
        open_analysis_action.triggered.connect(self.open_analysis_data_file)
        file_menu.addAction(open_analysis_action)
        save_analysis_action = QAction("Save Analysis Data As...", self)
        save_analysis_action.triggered.connect(self.save_analysis_data_as)
        file_menu.addAction(save_analysis_action)

    def _has_results_store(self):
        return bool(self.results_hdf5_path) and os.path.exists(self.results_hdf5_path)

    def _load_results_index(self):
        if not self._has_results_store():
            return pd.DataFrame(columns=['result_id', 'file', 'behavior', 'animal_id', 'date', 'time', 'datetime', 'session', 'max_peak', 'auc'])
        return hdf_store.load_results_index(self.results_hdf5_path)

    def _apply_results_store_metadata(self):
        if not self._has_results_store():
            return

        metadata = hdf_store.load_store_metadata(self.results_hdf5_path)
        event_prior = metadata.get('event_prior')
        event_follow = metadata.get('event_follow')

        if hasattr(self, 'event_prior_input') and event_prior not in (None, ''):
            self.event_prior_input.setText(str(event_prior))
        if hasattr(self, 'event_follow_input') and event_follow not in (None, ''):
            self.event_follow_input.setText(str(event_follow))

    def set_results_store(self, hdf5_path, success_message=None):
        self.results_hdf5_path = os.path.normpath(hdf5_path)
        self.plot_data = pd.DataFrame()
        self._apply_results_store_metadata()
        self.update_results_and_visualization_options()
        self.status_bar.showMessage(f"Analysis store ready: {self.results_hdf5_path}")
        if success_message:
            QMessageBox.information(self, "Analysis Data", success_message)

    def open_analysis_data_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Analysis Data", "", "HDF5 Files (*.hdf5 *.h5)")
        if not file_path:
            return

        try:
            hdf_store.load_store_metadata(file_path)
            self.set_results_store(file_path, f"Loaded analysis data from:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Open Analysis Data", f"Failed to open analysis data file:\n{e}")

    def save_analysis_data_as(self):
        if not self._has_results_store():
            QMessageBox.warning(self, "No Analysis Data", "Run an analysis or open an analysis data file first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Analysis Data As", "", "HDF5 Files (*.hdf5 *.h5)")
        if not file_path:
            return

        try:
            shutil.copy2(self.results_hdf5_path, file_path)
            QMessageBox.information(self, "Analysis Data", f"Analysis data saved to:\n{file_path}")
        except OSError as e:
            QMessageBox.critical(self, "Save Analysis Data", f"Failed to save analysis data:\n{e}")

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
        self.event_table.setColumnCount(6)
        self.event_table.setHorizontalHeaderLabels(['event_alias','event_type', 'event_name', 'event_group', 'event_arg', 'num_filter'])
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

                if (header == 'event_type' or header.startswith('filter_type')) and self.template_loaded:
                    suffix = header.replace('event_type', '').replace('filter_type', '')
                    t_name = 'event_type' if 'event_type' in header else 'filter_type' + suffix
                    n_name = 'event_name' if 'event_type' in header else 'filter_name' + suffix
                    g_name = 'event_group' if 'event_type' in header else 'filter_group' + suffix

                    event_type_combo = QComboBox()
                    event_type_combo.addItems(self.unique_event_types)
                    event_type_combo.setCurrentText(cell_val)
                    event_type_combo.currentTextChanged.connect(
                        lambda text, r=row, tn=t_name, nn=n_name, gn=g_name: 
                        self.filter_event_data(r, text, table_widget, type_col_name=tn, name_col_name=nn, group_col_name=gn)
                    )
                    item = QTableWidgetItem(cell_val)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    table_widget.setItem(row, col, item)
                    table_widget.setCellWidget(row, col, event_type_combo)

                elif (header == 'event_name' or header.startswith('filter_name')) and self.template_loaded:
                    suffix = header.replace('event_name', '').replace('filter_name', '')
                    t_name = 'event_type' if 'event_name' in header else 'filter_type' + suffix
                    n_name = 'event_name' if 'event_name' in header else 'filter_name' + suffix
                    g_name = 'event_group' if 'event_name' in header else 'filter_group' + suffix

                    event_name_combo = QComboBox()
                    event_name_combo.currentTextChanged.connect(
                        lambda text, r=row, tn=t_name, nn=n_name, gn=g_name: 
                        self.filter_event_group(r, text, table_widget, type_col_name=tn, name_col_name=nn, group_col_name=gn)
                    )
                    item = QTableWidgetItem(cell_val)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    table_widget.setItem(row, col, item)
                    table_widget.setCellWidget(row, col, event_name_combo)

                elif (header == 'event_group' or header.startswith('filter_group')) and self.template_loaded:
                    event_group_combo = QComboBox()
                    item = QTableWidgetItem(cell_val)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    table_widget.setItem(row, col, item)
                    table_widget.setCellWidget(row, col, event_group_combo)

                elif header.startswith('filter_prior') and self.template_loaded:
                    filter_prior_combo = QComboBox()
                    filter_prior_combo.addItems(['True', 'False'])
                    # Convert Binary 0/1 to True/False
                    if cell_val == '0':
                        cell_val = 'False'
                    elif cell_val == '1':
                        cell_val = 'True'
                    filter_prior_combo.setCurrentText(cell_val)
                    item = QTableWidgetItem(cell_val)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    table_widget.setItem(row, col, item)
                    table_widget.setCellWidget(row, col, filter_prior_combo)

                elif header == 'num_filter':
                    num_filter_spinbox = QSpinBox()
                    try:
                        f_val = int(float(cell_val)) if cell_val else 0
                    except ValueError:
                        f_val = 0
                    num_filter_spinbox.setValue(f_val)
                    num_filter_spinbox.valueChanged.connect(lambda value, r=row: self.adjust_filter_columns(r, value, table_widget))
                    
                    # Fix: set a non-editable item for the cell to prevent the table editor from interfering
                    # with the spinbox buttons.
                    item = QTableWidgetItem(str(f_val))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    table_widget.setItem(row, col, item)
                    table_widget.setCellWidget(row, col, num_filter_spinbox)

                elif header in ('abet_path', 'doric_path'):
                    fp_widget = FilePathWidget(cell_val)
                    item = QTableWidgetItem(cell_val)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    table_widget.setItem(row, col, item)
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
            # Initialize dropdown cascade for any filter_name{i} based on filter_type{i}
            # Calculate max value of init_vals['num_filter']
            if self.template_loaded and 'num_filter' in init_vals:
                try:
                    max_filter = int(float(init_vals['num_filter']))
                except (ValueError, TypeError):
                    max_filter = 0
                for i in range(1, max_filter + 1):
                    if f'filter_type{i}' in init_vals and f'filter_name{i}' in init_vals:
                        self.filter_event_data(row, init_vals[f'filter_type{i}'], table_widget, 
                                               initial_name=init_vals.get(f'filter_name{i}'), 
                                               initial_group=init_vals.get(f'filter_group{i}'),
                                               type_col_name=f'filter_type{i}',
                                               name_col_name=f'filter_name{i}',
                                               group_col_name=f'filter_group{i}')
        # Ensure columns and rows are sized to show content and refresh the widget so rows become visible
        table_widget.resizeColumnsToContents()
        table_widget.resizeRowsToContents()
        try:
            table_widget.scrollToTop()
        except Exception:
            pass
        table_widget.repaint()
        table_widget.setVisible(True)

    def _find_column(self, table_widget, header_name):
        """Return the column index whose header matches *header_name* (case-insensitive), or -1."""
        for col in range(table_widget.columnCount()):
            item = table_widget.horizontalHeaderItem(col)
            if item and item.text().lower() == header_name.lower():
                return col
        return -1

    def filter_event_data(self, row, event_type, table_widget, initial_name=None, initial_group=None, type_col_name='event_type', name_col_name='event_name', group_col_name='event_group'):
        if not self.template_loaded:
            return
        filtered_names = self.event_data[self.event_data['Evnt_Name'] == event_type]['Item_Name'].dropna().unique()
        name_col = self._find_column(table_widget, name_col_name)
        name_combo = table_widget.cellWidget(row, name_col) if name_col >= 0 else None
        if not isinstance(name_combo, QComboBox):
            return
            
        name_combo.blockSignals(True)
        name_combo.clear()
        name_combo.addItems(filtered_names)
        if initial_name and initial_name in filtered_names:
            name_combo.setCurrentText(initial_name)
        name_combo.blockSignals(False)
        
        current_name = name_combo.currentText()
        self.filter_event_group(row, current_name, table_widget, initial_group=initial_group, type_col_name=type_col_name, name_col_name=name_col_name, group_col_name=group_col_name)

    def filter_event_group(self, row, event_name, table_widget, initial_group=None, type_col_name='event_type', name_col_name='event_name', group_col_name='event_group'):
        if not self.template_loaded:
            return
        type_col = self._find_column(table_widget, type_col_name)
        event_type_combo = table_widget.cellWidget(row, type_col) if type_col >= 0 else None
        if not isinstance(event_type_combo, QComboBox):
            return
        event_type = event_type_combo.currentText()
        filtered_groups = self.event_data[(self.event_data['Evnt_Name'] == event_type) & (self.event_data['Item_Name'] == event_name)]['Group_ID'].dropna().unique()
        group_col = self._find_column(table_widget, group_col_name)
        group_combo = table_widget.cellWidget(row, group_col) if group_col >= 0 else None
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
            if (header == 'event_type' or header.startswith('filter_type')) and self.template_loaded:
                suffix = header.replace('event_type', '').replace('filter_type', '')
                t_name = 'event_type' if 'event_type' in header else 'filter_type' + suffix
                n_name = 'event_name' if 'event_type' in header else 'filter_name' + suffix
                g_name = 'event_group' if 'event_type' in header else 'filter_group' + suffix

                event_type_combo = QComboBox()
                event_type_combo.currentTextChanged.connect(
                    lambda text, r=current_row_count, tn=t_name, nn=n_name, gn=g_name: 
                    self.filter_event_data(r, text, table_widget, type_col_name=tn, name_col_name=nn, group_col_name=gn)
                )
                event_type_combo.addItems(self.unique_event_types)
                item = QTableWidgetItem("")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table_widget.setItem(current_row_count, col, item)
                table_widget.setCellWidget(current_row_count, col, event_type_combo)

            elif (header == 'event_name' or header.startswith('filter_name')) and self.template_loaded:
                suffix = header.replace('event_name', '').replace('filter_name', '')
                t_name = 'event_type' if 'event_name' in header else 'filter_type' + suffix
                n_name = 'event_name' if 'event_name' in header else 'filter_name' + suffix
                g_name = 'event_group' if 'event_name' in header else 'filter_group' + suffix

                event_name_combo = QComboBox()
                event_name_combo.currentTextChanged.connect(
                    lambda text, r=current_row_count, tn=t_name, nn=n_name, gn=g_name: 
                    self.filter_event_group(r, text, table_widget, type_col_name=tn, name_col_name=nn, group_col_name=gn)
                )
                item = QTableWidgetItem("")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table_widget.setItem(current_row_count, col, item)
                table_widget.setCellWidget(current_row_count, col, event_name_combo)

            elif (header == 'event_group' or header.startswith('filter_group')) and self.template_loaded:
                event_group_combo = QComboBox()
                item = QTableWidgetItem("")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table_widget.setItem(current_row_count, col, item)
                table_widget.setCellWidget(current_row_count, col, event_group_combo)

            elif header == 'num_filter':
                num_filter_spinbox = QSpinBox()
                num_filter_spinbox.setValue(0)
                num_filter_spinbox.valueChanged.connect(lambda value, r=current_row_count: self.adjust_filter_columns(r, value, table_widget))
                item = QTableWidgetItem("0")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table_widget.setItem(current_row_count, col, item)
                table_widget.setCellWidget(current_row_count, col, num_filter_spinbox)

            elif header in ('abet_path', 'doric_path'):
                fp_widget = FilePathWidget()
                item = QTableWidgetItem("")
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table_widget.setItem(current_row_count, col, item)
                table_widget.setCellWidget(current_row_count, col, fp_widget)
            else:
                item = QTableWidgetItem("")
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                table_widget.setItem(current_row_count, col, item)

        # Initialize dropdowns for the new row if template is loaded
        if self.template_loaded:
            # Initialize main event
            type_col = self._find_column(table_widget, 'event_type')
            type_combo = table_widget.cellWidget(current_row_count, type_col) if type_col >= 0 else None
            if isinstance(type_combo, QComboBox) and type_combo.count() > 0:
                self.filter_event_data(current_row_count, type_combo.currentText(), table_widget)
            
            # Initialize any filter dropdowns and ensure they match num_filter=0 state
            num_filter_col = self._find_column(table_widget, 'num_filter')
            num_filters = 0
            if num_filter_col >= 0:
                spin = table_widget.cellWidget(current_row_count, num_filter_col)
                if isinstance(spin, QSpinBox):
                    num_filters = spin.value()
            
            # Always call adjust_filter_columns to ensure proper state for the new row
            self.adjust_filter_columns(current_row_count, num_filters, table_widget)
            
            # If for some reason num_filters > 0, initialize cascades
            for i in range(1, num_filters + 1):
                f_type_col = self._find_column(table_widget, f'filter_type{i}')
                f_type_combo = table_widget.cellWidget(current_row_count, f_type_col) if f_type_col >= 0 else None
                if isinstance(f_type_combo, QComboBox) and f_type_combo.count() > 0:
                    self.filter_event_data(current_row_count, f_type_combo.currentText(), table_widget,
                                           type_col_name=f'filter_type{i}',
                                           name_col_name=f'filter_name{i}',
                                           group_col_name=f'filter_group{i}')

    def adjust_filter_columns(self, row, num_filters, table_widget):
        # Determine fixed-column count dynamically so sheets with or without
        # event_alias (and any other future leading columns) are handled correctly.
        num_filter_col = self._find_column(table_widget, 'num_filter')
        fixed_cols = (num_filter_col + 1) if num_filter_col >= 0 else 6
        
        # Calculate maximum needed columns across all rows to avoid truncating data
        max_filters = 0
        for r in range(table_widget.rowCount()):
            spin = table_widget.cellWidget(r, num_filter_col)
            if isinstance(spin, QSpinBox):
                max_filters = max(max_filters, spin.value())
        
        total_columns = fixed_cols + (max_filters * 6)
        table_widget.setColumnCount(total_columns)
        
        for i in range(1, max_filters + 1):
            base_index = fixed_cols + (i - 1) * 6
            headers = [f'filter_type{i}', f'filter_name{i}', f'filter_group{i}', f'filter_arg{i}', f'filter_eval{i}', f'filter_prior{i}']
            for j, header in enumerate(headers):
                if table_widget.horizontalHeaderItem(base_index + j) is None:
                    table_widget.setHorizontalHeaderItem(base_index + j, QTableWidgetItem(header))
                
                # Only initialize widgets for the current row's active filters
                if i <= num_filters:
                    h_lower = header.lower()
                    if f'filter_type{i}' in h_lower and self.template_loaded:
                        t_name, n_name, g_name = f'filter_type{i}', f'filter_name{i}', f'filter_group{i}'
                        combo = QComboBox()
                        combo.addItems(self.unique_event_types)
                        combo.currentTextChanged.connect(
                            lambda text, r=row, tn=t_name, nn=n_name, gn=g_name: 
                            self.filter_event_data(r, text, table_widget, type_col_name=tn, name_col_name=nn, group_col_name=gn)
                        )
                        table_widget.setCellWidget(row, base_index + j, combo)
                        # Initialize next in cascade
                        self.filter_event_data(row, combo.currentText(), table_widget, type_col_name=t_name, name_col_name=n_name, group_col_name=g_name)
                    elif f'filter_name{i}' in h_lower and self.template_loaded:
                        t_name, n_name, g_name = f'filter_type{i}', f'filter_name{i}', f'filter_group{i}'
                        combo = QComboBox()
                        combo.currentTextChanged.connect(
                            lambda text, r=row, tn=t_name, nn=n_name, gn=g_name: 
                            self.filter_event_group(r, text, table_widget, type_col_name=tn, name_col_name=nn, group_col_name=gn)
                        )
                        table_widget.setCellWidget(row, base_index + j, combo)
                    elif f'filter_group{i}' in h_lower and self.template_loaded:
                        combo = QComboBox()
                        table_widget.setCellWidget(row, base_index + j, combo)

                    # Set filter_prior{i} to a ComboBox with values True and False corresponding to 1 and 0
                    elif f'filter_prior{i}' in h_lower and self.template_loaded:
                        combo = QComboBox()
                        combo.addItems(["True", "False"])
                        table_widget.setCellWidget(row, base_index + j, combo)
                        combo.setCurrentText("True")
                    
                    # Ensure item exists and is non-editable for widget columns
                    item = table_widget.item(row, base_index + j)
                    if item is None:
                        item = QTableWidgetItem("")
                        table_widget.setItem(row, base_index + j, item)
                    if h_lower.startswith('filter_type') or h_lower.startswith('filter_name') or h_lower.startswith('filter_group'):
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                else:
                    # Clear widget if filter is no longer active for this row
                    table_widget.setCellWidget(row, base_index + j, None)
                    item = table_widget.item(row, base_index + j)
                    if item is None:
                        item = QTableWidgetItem("")
                        table_widget.setItem(row, base_index + j, item)
                    item.setFlags(item.flags() | Qt.ItemIsEditable)

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
                    elif isinstance(widget, QSpinBox):
                        row_data.append(str(widget.value()))
                    elif isinstance(widget, QCheckBox):
                        row_data.append("True" if widget.isChecked() else "False")
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

    def _on_analysis_complete(self, hdf5_path):
        """Called on the main thread when the analysis worker finishes."""
        self.run_analysis_button.setEnabled(True)
        self.run_analysis_button.setText("Run Analysis")
        self.progress_bar.hide()
        self.status_bar.showMessage("Analysis Complete")
        self.set_results_store(hdf5_path, f"Analysis complete. Results were written to:\n{hdf5_path}")

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
            headers = []
            for col in range(cols):
                header_item = self.results_table.horizontalHeaderItem(col)
                headers.append(header_item.text() if header_item else f"Column {col + 1}")
            for row in range(rows):
                row_data = []
                for col in range(cols):
                    item = self.results_table.item(row, col)
                    row_data.append(item.text() if item else '')
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
        self.vis_date_mode.addItems(["Date", "Date Time", "Order", "First/Last"])
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
        full_screen_button = QPushButton("Open Full Screen")
        full_screen_button.clicked.connect(self.open_full_screen_plot)
        save_layout.addWidget(save_data_button)
        save_layout.addWidget(save_graph_button)
        save_layout.addWidget(full_screen_button)
        
        layout.addLayout(save_layout)
        self.visualization_tab.setLayout(layout)

    def _result_index_with_animals(self):
        index_df = self._load_results_index().copy()
        if index_df.empty:
            return index_df

        for column in ['result_id', 'file', 'behavior', 'animal_id', 'date', 'time', 'datetime', 'session']:
            if column in index_df:
                index_df[column] = index_df[column].fillna('').astype(str)

        return index_df[index_df['animal_id'].str.len() > 0].copy()

    def _sort_selector_values(self, values, mode='Date'):
        unique_values = {str(value) for value in values if str(value)}
        if mode == 'Order':
            def sort_key(value):
                try:
                    return (0, int(str(value).split()[-1]), str(value))
                except (ValueError, IndexError):
                    return (1, str(value), str(value))
        else:
            def sort_key(value):
                parsed = pd.to_datetime(str(value), errors='coerce')
                if pd.isna(parsed):
                    return (1, str(value), str(value))
                return (0, parsed.value, str(value))

        return [item for item in sorted(unique_values, key=sort_key)]

    def _build_first_last_map(self, index_df):
        first_last_map = {}
        if index_df.empty:
            return first_last_map

        working_df = index_df.copy()
        working_df['first_last_key'] = working_df['datetime'].where(working_df['datetime'].str.len() > 0, working_df['date'])

        for animal_id, animal_df in working_df.groupby('animal_id'):
            ordered_labels = self._sort_selector_values(animal_df['first_last_key'].tolist(), mode='Date Time')
            first_last_map[str(animal_id)] = {
                'First': ordered_labels[0] if ordered_labels else None,
                'Last': ordered_labels[-1] if ordered_labels else None,
            }

        return first_last_map

    def update_animal_date_selects(self):
        """Populate animal and date selectors from the persisted HDF5 result index."""
        index_df = self._result_index_with_animals()
        animals = self._sort_selector_values(index_df['animal_id'].tolist()) if not index_df.empty else []

        # Update Results selectors if present
        if hasattr(self, 'results_animal_select') and hasattr(self, 'results_date_select'):
            self.results_animal_select.blockSignals(True)
            self.results_date_select.blockSignals(True)
            self.results_animal_select.clear()
            self.results_animal_select.addItem('Combine')
            self.results_animal_select.addItem('All')
            for a in animals:
                self.results_animal_select.addItem(str(a))
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
            for a in animals:
                self.vis_animal_select.add_option(str(a))
                
            self.vis_date_select.list_widget.clear()
            self.update_vis_date_select()

    def update_vis_date_select(self, *_):
        """Populate the visualization date select when the animal selection changes."""
        self.vis_date_select.list_widget.clear()
        selected_animals = self.vis_animal_select.get_checked_items() if hasattr(self, 'vis_animal_select') else []
        index_df = self._result_index_with_animals()
        if not selected_animals or index_df.empty:
            return
        
        mode = self.vis_date_mode.currentText() if hasattr(self, 'vis_date_mode') else "Date"

        if mode == "First/Last":
            self.vis_date_select.add_option('All')
            self.vis_date_select.add_option('First')
            self.vis_date_select.add_option('Last')
            self.update_vis_behavior_select()
            return

        if 'All' not in selected_animals:
            index_df = index_df[index_df['animal_id'].isin(selected_animals)]

        if mode == "Date":
            date_values = index_df['date'].tolist()
        elif mode == "Date Time":
            date_values = index_df['datetime'].where(index_df['datetime'].str.len() > 0, index_df['date']).tolist()
        else:
            date_values = index_df['session'].where(index_df['session'].str.len() > 0, index_df['date']).tolist()

        # allow viewing all dates for the animal by adding options
        self.vis_date_select.add_option('All')
        sorted_dates = self._sort_selector_values(date_values, mode=mode)
            
        for d in sorted_dates:
            self.vis_date_select.add_option(str(d))
        self.update_vis_behavior_select()

    def update_results_date_select(self, animal: str):
        """Populate the results date select when the results animal selection changes."""
        self.results_date_select.clear()
        index_df = self._result_index_with_animals()
        if not animal or index_df.empty:
            return
        
        if animal in ['Combine', 'All']:
            date_values = index_df['date'].tolist()
        else:
            date_values = index_df.loc[index_df['animal_id'] == animal, 'date'].tolist()
        dates = self._sort_selector_values(date_values)
            
        self.results_date_select.addItem('Combine')
        self.results_date_select.addItem('All')
        for d in dates:
            self.results_date_select.addItem(str(d))
        self.update_results_behavior_select()

    def update_results_behavior_select(self, *_):
        """Populate the results behavior select based on animal and date selections."""
        self.results_behavior_select.clear()
        selected_animal = self.results_animal_select.currentText() if hasattr(self, 'results_animal_select') else ''
        selected_date = self.results_date_select.currentText() if hasattr(self, 'results_date_select') else ''
        index_df = self._result_index_with_animals()
        
        if not selected_animal or index_df.empty:
            return

        if selected_animal not in ['Combine', 'All']:
            index_df = index_df[index_df['animal_id'] == selected_animal]
        if selected_date and selected_date not in ['Combine', 'All']:
            index_df = index_df[index_df['date'] == selected_date]

        behaviors = self._sort_selector_values(index_df['behavior'].tolist())
        self.results_behavior_select.addItem('Combine')
        self.results_behavior_select.addItem('All')
        for b in behaviors:
            self.results_behavior_select.addItem(str(b))
        self.update_results_table()

    def update_results_table(self):
        selected_animal = self.results_animal_select.currentText() if hasattr(self, 'results_animal_select') else ''
        selected_date = self.results_date_select.currentText() if hasattr(self, 'results_date_select') else ''
        selected_behavior = self.results_behavior_select.currentText() if hasattr(self, 'results_behavior_select') else ''
        index_df = self._result_index_with_animals()
        
        if not selected_animal or index_df.empty:
            self.results_table.clear()
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            return

        if selected_animal not in ['Combine', 'All']:
            index_df = index_df[index_df['animal_id'] == selected_animal]
        if selected_date and selected_date not in ['Combine', 'All']:
            index_df = index_df[index_df['date'] == selected_date]
        if selected_behavior and selected_behavior not in ['Combine', 'All']:
            index_df = index_df[index_df['behavior'] == selected_behavior]

        working_df = index_df.copy()
        working_df['grp_animal'] = 'Combine' if selected_animal == 'Combine' else working_df['animal_id']
        working_df['grp_date'] = 'Combine' if selected_date == 'Combine' else working_df['date']
        working_df['grp_time'] = 'Combine' if selected_date == 'Combine' else working_df['time']
        working_df['grp_behavior'] = 'Combine' if selected_behavior == 'Combine' else working_df['behavior']

        grouped_results = working_df.groupby(
            ['grp_animal', 'grp_date', 'grp_time', 'grp_behavior'],
            dropna=False,
        ).agg(max_peak=('max_peak', 'mean'), auc=('auc', 'mean')).reset_index()

        # Determine dynamic columns
        include_animal = (selected_animal != 'Combine')
        include_date = (selected_date != 'Combine')
        include_behavior = (selected_behavior != 'Combine')
        
        columns = []
        if include_animal: columns.append('Animal')
        if include_date:
            columns.append('Date')
            columns.append('Time')
        if include_behavior: columns.append('Behavior')
        columns.extend(['Max Peak', 'AUC'])

        self.results_table.setRowCount(len(grouped_results.index))
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        
        for row_number, (_, row) in enumerate(grouped_results.iterrows()):
            col_idx = 0
            grp_animal = str(row['grp_animal'])
            grp_date = str(row['grp_date'])
            grp_time = str(row['grp_time'])
            grp_behavior = str(row['grp_behavior'])
            
            if include_animal:
                self.results_table.setItem(row_number, col_idx, QTableWidgetItem(grp_animal))
                col_idx += 1
            if include_date:
                self.results_table.setItem(row_number, col_idx, QTableWidgetItem(grp_date))
                col_idx += 1
                self.results_table.setItem(row_number, col_idx, QTableWidgetItem(grp_time))
                col_idx += 1
            if include_behavior:
                self.results_table.setItem(row_number, col_idx, QTableWidgetItem(grp_behavior))
                col_idx += 1

            avg_max_peak = row['max_peak']
            avg_auc = row['auc']
            
            self.results_table.setItem(row_number, col_idx, QTableWidgetItem(f"{avg_max_peak:.4f}" if pd.notna(avg_max_peak) else 'N/A'))
            col_idx += 1
            self.results_table.setItem(row_number, col_idx, QTableWidgetItem(f"{avg_auc:.4f}" if pd.notna(avg_auc) else 'N/A'))
            
        self.results_table.resizeColumnsToContents()

    def update_vis_file_select(self):
        # legacy helper left for compatibility; delegate to animal/date selector updater
        self.update_animal_date_selects()

    def update_vis_behavior_select(self, *_):
        self.vis_behavior_select.list_widget.clear()
        selected_animals = self.vis_animal_select.get_checked_items() if hasattr(self, 'vis_animal_select') else []
        selected_dates = self.vis_date_select.get_checked_items() if hasattr(self, 'vis_date_select') else []
        index_df = self._result_index_with_animals()
        
        if not selected_animals or not selected_dates or index_df.empty:
            return

        mode = self.vis_date_mode.currentText() if hasattr(self, 'vis_date_mode') else "Date"

        if 'All' not in selected_animals:
            index_df = index_df[index_df['animal_id'].isin(selected_animals)]

        if mode == "First/Last":
            first_last_map = self._build_first_last_map(index_df)
            first_last_keys = index_df['datetime'].where(index_df['datetime'].str.len() > 0, index_df['date'])
            mask = pd.Series(False, index=index_df.index)
            if 'All' in selected_dates or 'First' in selected_dates:
                mask = mask | (first_last_keys == index_df['animal_id'].map(lambda animal: first_last_map.get(str(animal), {}).get('First')))
            if 'All' in selected_dates or 'Last' in selected_dates:
                mask = mask | (first_last_keys == index_df['animal_id'].map(lambda animal: first_last_map.get(str(animal), {}).get('Last')))
            behaviors = self._sort_selector_values(index_df.loc[mask, 'behavior'].tolist())
            self.vis_behavior_select.add_option('All')
            for b in behaviors:
                self.vis_behavior_select.add_option(str(b))
            return

        if mode == "Date":
            date_values = index_df['date']
        elif mode == "Date Time":
            date_values = index_df['datetime'].where(index_df['datetime'].str.len() > 0, index_df['date'])
        else:
            date_values = index_df['session'].where(index_df['session'].str.len() > 0, index_df['date'])

        if 'All' not in selected_dates:
            index_df = index_df[date_values.isin(selected_dates)]

        behaviors = self._sort_selector_values(index_df['behavior'].tolist())
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

        index_df = self._result_index_with_animals()
        if index_df.empty:
            QMessageBox.warning(self, "No Analysis Data", "Run an analysis or open an analysis data file first.")
            return

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

        working_df = index_df[index_df['animal_id'].isin(selected_animals)].copy()
        working_df = working_df[working_df['behavior'].isin(selected_behaviors)]

        if mode == "First/Last":
            first_last_map = self._build_first_last_map(working_df)
            display_date = working_df['datetime'].where(working_df['datetime'].str.len() > 0, working_df['date'])
            working_df['selected_date_value'] = ''
            if 'First' in selected_dates:
                first_mask = display_date == working_df['animal_id'].map(lambda animal: first_last_map.get(str(animal), {}).get('First'))
                working_df.loc[first_mask, 'selected_date_value'] = 'First'
            if 'Last' in selected_dates:
                last_mask = display_date == working_df['animal_id'].map(lambda animal: first_last_map.get(str(animal), {}).get('Last'))
                working_df.loc[last_mask, 'selected_date_value'] = 'Last'
            working_df = working_df[working_df['selected_date_value'].str.len() > 0]
        elif mode == "Date":
            working_df['selected_date_value'] = working_df['date']
            working_df = working_df[working_df['selected_date_value'].isin(selected_dates)]
        elif mode == "Date Time":
            working_df['selected_date_value'] = working_df['datetime'].where(working_df['datetime'].str.len() > 0, working_df['date'])
            working_df = working_df[working_df['selected_date_value'].isin(selected_dates)]
        else:
            working_df['selected_date_value'] = working_df['session'].where(working_df['session'].str.len() > 0, working_df['date'])
            working_df = working_df[working_df['selected_date_value'].isin(selected_dates)]

        grouped_raw_results = {}
        plot_data_map = hdf_store.load_plot_data_map(self.results_hdf5_path, working_df['result_id'].tolist())

        for _, res in working_df.iterrows():
            a = str(res['animal_id'])
            b = str(res['behavior'])
            d = str(res['selected_date_value'])
            plot_df = plot_data_map.get(str(res['result_id']), pd.DataFrame())

            if plot_df.empty:
                continue
                
            # Apply combination treatments immediately internally
            eff_a = 'Combined' if animal_treatment == "Combine" else a
            eff_d = 'All' if date_treatment == "Combine" else d
            eff_b = 'Combined' if behavior_treatment == "Combine" else b
            
            key = (eff_a, eff_d, eff_b)
            if key not in grouped_raw_results:
                grouped_raw_results[key] = []
            grouped_raw_results[key].append(plot_df)

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
        
        for res in matched_results:
            # Create a tuple representing which subplot this result belongs to
            sp_key = []
            if animal_treatment == "Separate Subplots": sp_key.append(res['animal'])
            if date_treatment == "Separate Subplots": sp_key.append(res['date'])
            if behavior_treatment == "Separate Subplots": sp_key.append(res['behavior'])
            
            sp_key = tuple(sp_key)
            if sp_key not in subplot_groups:
                subplot_groups[sp_key] = []
            subplot_groups[sp_key].append(res)
            
        subplots_needed = max(1, len(subplot_groups))
        
        # Create columns/rows for subplots
        cols = 1
        if subplots_needed > 1:
            cols = 2 if subplots_needed % 2 == 0 or subplots_needed > 3 else 1
        rows = (subplots_needed + cols - 1) // cols
        plot_axes = []
        
        for idx, (sp_key, Group) in enumerate(subplot_groups.items()):
            ax = self.canvas.figure.add_subplot(rows, cols, idx + 1)
            plot_axes.append(ax)
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
                               extent=(-event_prior, event_follow, data.shape[0], 0))
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
        for ax in plot_axes:
            if hasattr(self, 'vis_y_min') and self.vis_y_min.text() and vis_mode != 'heatmap':
                try:
                    ax.set_ylim(bottom=float(self.vis_y_min.text()))
                except ValueError:
                    pass
            if hasattr(self, 'vis_y_max') and self.vis_y_max.text() and vis_mode != 'heatmap':
                try:
                    ax.set_ylim(top=float(self.vis_y_max.text()))
                except ValueError:
                    pass
            if hasattr(self, 'vis_x_min') and self.vis_x_min.text():
                try:
                    ax.set_xlim(left=float(self.vis_x_min.text()))
                except ValueError:
                    pass
            if hasattr(self, 'vis_x_max') and self.vis_x_max.text():
                try:
                    ax.set_xlim(right=float(self.vis_x_max.text()))
                except ValueError:
                    pass

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

    def open_full_screen_plot(self):
        if not self.canvas.figure.axes:
            QMessageBox.warning(self, "No Plot", "Generate a plot first.")
            return

        try:
            self.plot_window = PlotViewerWindow(self.canvas.figure, self)
            self.plot_window.show()
            self.plot_window.raise_()
            self.plot_window.activateWindow()
        except Exception as e:
            QMessageBox.critical(self, "Plot Viewer", f"Failed to open full-screen plot window:\n{e}")

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
            robust_fit_widgets = []
            despike_param_widgets = []
            despike_checkbox_ref = [None]

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

                # File path keys: line edit + browse button
                elif key in file_path_keys:
                    hbox = QHBoxLayout()
                    line_edit = QLineEdit(value)
                    browse_btn = QPushButton("Browse...")

                    def make_browse_handler(le=line_edit, k=key, dk=display_key):
                        def handler():
                            mode = file_path_keys.get(k, 'open')
                            if mode == 'open':
                                file_path, _ = QFileDialog.getOpenFileName(self, f"Select {dk}", "", "All Files (*)")
                                if file_path:
                                    le.setText(file_path)
                            elif mode == 'directory':
                                dir_path = QFileDialog.getExistingDirectory(self, f"Select {dk}")
                                if dir_path:
                                    le.setText(dir_path)
                            else:
                                file_path, _ = QFileDialog.getOpenFileName(self, f"Select {dk}", "", "All Files (*)")
                                if file_path:
                                    le.setText(file_path)
                        return handler

                    browse_btn.clicked.connect(make_browse_handler())
                    hbox.addWidget(line_edit)
                    hbox.addWidget(browse_btn)
                    setattr(self, f"{widget_attr_base}_line_edit", line_edit)
                    group_layout.addRow(display_key, hbox)

                # filter_type combobox
                elif key == 'filter_type':
                    combo = QComboBox()
                    combo.addItems(['lowpass', 'smoothing'])
                    combo.setCurrentText(value)
                    group_layout.addRow(display_key, combo)
                    setattr(self, f"{widget_attr_base}_combobox", combo)

                # filter_name combobox — options depend on filter_type
                elif key == 'filter_name':
                    combo = QComboBox()
                    ft_attr = f"{section}_filter_type_combobox"
                    ft_combo = getattr(self, ft_attr, None)

                    def update_filter_name_options(ftc=ft_combo, c=combo, v=value):
                        ftype = ftc.currentText() if ftc else 'lowpass'
                        c.clear()
                        if ftype == 'lowpass':
                            c.addItems(['butter', 'bessel', 'chebychev'])
                        else:
                            c.addItem('savitsky-golay')
                        c.setCurrentText(v)

                    if ft_combo:
                        ft_combo.currentTextChanged.connect(
                            lambda _, ftc=ft_combo, c=combo, v=value: update_filter_name_options(ftc, c, v))
                    update_filter_name_options()
                    group_layout.addRow(display_key, combo)
                    setattr(self, f"{widget_attr_base}_combobox", combo)

                # filter_order / filter_cutoff: only shown for lowpass
                elif key in ('filter_order', 'filter_cutoff'):
                    ft_attr = f"{section}_filter_type_combobox"
                    ft_combo = getattr(self, ft_attr, None)
                    spin_box = QSpinBox()
                    if key == 'filter_cutoff':
                        spin_box.setRange(1, 1000)
                        default_value = int(value) if value.isdigit() else 10
                    else:
                        spin_box.setRange(1, 10)
                        default_value = int(value) if value.isdigit() else 4
                    spin_box.setValue(default_value)
                    label = QLabel(display_key)
                    group_layout.addRow(label, spin_box)
                    spin_box.setVisible(False)
                    label.setVisible(False)
                    setattr(self, f"{widget_attr_base}_spin_box", spin_box)

                    def update_lowpass_vis(ftc=ft_combo, sb=spin_box, lbl=label):
                        visible = ftc is not None and ftc.currentText() == 'lowpass'
                        sb.setVisible(visible)
                        lbl.setVisible(visible)

                    if ft_combo:
                        ft_combo.currentTextChanged.connect(
                            lambda _, ftc=ft_combo, sb=spin_box, lbl=label: update_lowpass_vis(ftc, sb, lbl))
                    update_lowpass_vis()

                # savgol_window / savgol_polyorder: only shown when filter_name == savitsky-golay
                elif key in ('savgol_window', 'savgol_polyorder'):
                    fn_attr = f"{section}_filter_name_combobox"
                    fn_combo = getattr(self, fn_attr, None)
                    spin_box = QSpinBox()
                    try:
                        spin_box.setValue(int(value))
                    except ValueError:
                        spin_box.setValue(3)
                    label = QLabel(display_key)
                    group_layout.addRow(label, spin_box)
                    spin_box.setVisible(False)
                    label.setVisible(False)
                    setattr(self, f"{widget_attr_base}_spin_box", spin_box)

                    def update_savgol_vis(fnc=fn_combo, sb=spin_box, lbl=label):
                        visible = fnc is not None and fnc.currentText() == 'savitsky-golay'
                        sb.setVisible(visible)
                        lbl.setVisible(visible)

                    if fn_combo:
                        fn_combo.currentTextChanged.connect(
                            lambda _, fnc=fn_combo, sb=spin_box, lbl=label: update_savgol_vis(fnc, sb, lbl))
                    update_savgol_vis()

                # cheby_ripple: float QLineEdit, visible only when filter_type==lowpass AND filter_name==chebychev
                elif key == 'cheby_ripple':
                    fn_attr = f"{section}_filter_name_combobox"
                    ft_attr = f"{section}_filter_type_combobox"
                    fn_combo = getattr(self, fn_attr, None)
                    ft_combo = getattr(self, ft_attr, None)
                    ripple_edit = QLineEdit(value)
                    label = QLabel(display_key)
                    group_layout.addRow(label, ripple_edit)
                    ripple_edit.setVisible(False)
                    label.setVisible(False)
                    setattr(self, f"{widget_attr_base}_line_edit", ripple_edit)

                    def update_cheby_vis(fnc=fn_combo, ftc=ft_combo, w=ripple_edit, lbl=label):
                        visible = (fnc is not None and ftc is not None and
                                   fnc.currentText() == 'chebychev' and
                                   ftc.currentText() == 'lowpass')
                        w.setVisible(visible)
                        lbl.setVisible(visible)

                    if fn_combo:
                        fn_combo.currentTextChanged.connect(
                            lambda _, fnc=fn_combo, ftc=ft_combo, w=ripple_edit, lbl=label: update_cheby_vis(fnc, ftc, w, lbl))
                    if ft_combo:
                        ft_combo.currentTextChanged.connect(
                            lambda _, fnc=fn_combo, ftc=ft_combo, w=ripple_edit, lbl=label: update_cheby_vis(fnc, ftc, w, lbl))
                    update_cheby_vis()

                # despike: checkbox — despike_window / despike_threshold depend on this
                elif key == 'despike':
                    checkbox = QCheckBox()
                    checkbox.setChecked(value.lower() in ('true', '1'))
                    group_layout.addRow(display_key, checkbox)
                    setattr(self, f"{widget_attr_base}_checkbox", checkbox)
                    despike_checkbox_ref[0] = checkbox

                # despike_window: integer spinbox, only visible when despike is checked
                elif key == 'despike_window':
                    spin_box = QSpinBox()
                    spin_box.setRange(3, 100001)
                    spin_box.setSingleStep(2)
                    try:
                        spin_box.setValue(int(value))
                    except ValueError:
                        spin_box.setValue(2001)
                    label = QLabel(display_key)
                    group_layout.addRow(label, spin_box)
                    spin_box.setVisible(False)
                    label.setVisible(False)
                    setattr(self, f"{widget_attr_base}_spin_box", spin_box)
                    despike_param_widgets.append((label, spin_box))

                # despike_threshold: float QLineEdit, only visible when despike is checked
                elif key == 'despike_threshold':
                    threshold_edit = QLineEdit(value)
                    label = QLabel(display_key)
                    group_layout.addRow(label, threshold_edit)
                    threshold_edit.setVisible(False)
                    label.setVisible(False)
                    setattr(self, f"{widget_attr_base}_line_edit", threshold_edit)
                    despike_param_widgets.append((label, threshold_edit))

                # robust_fit: checkbox, only visible when fit_type == linear
                elif key == 'robust_fit':
                    checkbox = QCheckBox()
                    checkbox.setChecked(value.lower() in ('true', '1'))
                    label = QLabel(display_key)
                    group_layout.addRow(label, checkbox)
                    setattr(self, f"{widget_attr_base}_checkbox", checkbox)
                    robust_fit_widgets.append((label, checkbox))

                # huber_epsilon: only visible when fit_type == linear (grouped with robust_fit)
                elif key == 'huber_epsilon':
                    huber_edit = QLineEdit(value)
                    huber_edit.setToolTip("'auto' or 'mad' = calculate from MAD of noise floor; or enter a numeric value (must be > 1.0)")
                    label = QLabel("Huber Epsilon:")
                    group_layout.addRow(label, huber_edit)
                    huber_edit.setVisible(False)
                    label.setVisible(False)
                    setattr(self, f"{widget_attr_base}_line_edit", huber_edit)
                    robust_fit_widgets.append((label, huber_edit))

                # ARPLS options: all collected for show/hide; booleans get checkboxes, others get line edits
                elif key.startswith('arpls_'):
                    if value.lower() in ('true', 'false', '1', '0'):
                        arpls_widget = QCheckBox()
                        arpls_widget.setChecked(value.lower() in ('true', '1'))
                        setattr(self, f"{widget_attr_base}_checkbox", arpls_widget)
                    else:
                        arpls_widget = QLineEdit(value)
                        setattr(self, f"{widget_attr_base}_line_edit", arpls_widget)
                    label = QLabel(display_key)
                    group_layout.addRow(label, arpls_widget)
                    arpls_widgets.append((label, arpls_widget))

                # Generic boolean values (except Output section) get checkboxes
                elif value.lower() in ('true', 'false', '1', '0') and section != "Output":
                    checkbox = QCheckBox()
                    checkbox.setChecked(value.lower() in ('true', '1'))
                    group_layout.addRow(display_key, checkbox)
                    setattr(self, f"{widget_attr_base}_checkbox", checkbox)

                # Everything else: plain QLineEdit
                else:
                    line_edit = QLineEdit(value)
                    setattr(self, f"{widget_attr_base}_line_edit", line_edit)
                    group_layout.addRow(display_key, line_edit)

            # Connect arpls_ widget visibility to fit_type (only shown for arpls)
            if fit_type_combo and arpls_widgets:
                def update_arpls_visibility(ftc=fit_type_combo, widgets=list(arpls_widgets)):
                    is_arpls = ftc.currentText() == 'arpls'
                    for lbl, w in widgets:
                        lbl.setVisible(is_arpls)
                        w.setVisible(is_arpls)
                fit_type_combo.currentTextChanged.connect(
                    lambda _, ftc=fit_type_combo, widgets=list(arpls_widgets): update_arpls_visibility(ftc, widgets))
                update_arpls_visibility()

            # Connect robust_fit visibility to fit_type (only shown when linear)
            if fit_type_combo and robust_fit_widgets:
                def update_robust_visibility(ftc=fit_type_combo, widgets=list(robust_fit_widgets)):
                    visible = ftc.currentText() == 'linear'
                    for lbl, w in widgets:
                        lbl.setVisible(visible)
                        w.setVisible(visible)
                fit_type_combo.currentTextChanged.connect(
                    lambda _, ftc=fit_type_combo, widgets=list(robust_fit_widgets): update_robust_visibility(ftc, widgets))
                update_robust_visibility()

            # Connect despike_window / despike_threshold visibility to despike checkbox
            if despike_checkbox_ref[0] and despike_param_widgets:
                def update_despike_param_visibility(cb=despike_checkbox_ref[0], widgets=list(despike_param_widgets)):
                    visible = cb.isChecked()
                    for lbl, w in widgets:
                        lbl.setVisible(visible)
                        w.setVisible(visible)
                despike_checkbox_ref[0].stateChanged.connect(
                    lambda _, cb=despike_checkbox_ref[0], widgets=list(despike_param_widgets): update_despike_param_visibility(cb, widgets))
                update_despike_param_visibility()

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