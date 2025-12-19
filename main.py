import sys 
import numpy as np
from scipy.signal import butter,filtfilt,spectrogram
from scipy.fft import fft, fftfreq
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QSlider, QTabWidget)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import sounddevice as sd
import soundfile as sf


def apply_window_fixed(signal, window_type):
    N = len(signal)
    if window_type == "Hann":
        w = np.hanning(N)
    elif window_type == "Hamming":
        w = np.hamming(N)
    elif window_type == "Blackman":
        w = np.blackman(N)
    else:
        w = np.ones(N)
    return signal * w, w



def butter_filter(signal, sr, cutoff, ftype):
    b, a = butter(4, cutoff / (sr / 2), btype=ftype)
    return filtfilt(b, a, signal)


def compute_fft(signal, sr, window):
    signal = signal - np.mean(signal)
    N = len(signal)
    T = 1 / sr
    yf = fft(signal)
    xf = fftfreq(N, T)[:N // 2]
    magnitude = np.abs(yf[:N // 2]) * 2 / np.sum(window)
    return xf, magnitude


def compute_spectrogram(signal, sr):
    signal = signal - np.mean(signal)
    f, t, Sxx = spectrogram(signal, sr, nperseg=1024, noverlap=512)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    return f, t, Sxx_db





class FourierAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Analyzer - Audio Signal Processing")
        self.resize(1200, 800)

        self.signal = None
        self.filtered = None
        self.sr = None
        self.window_type = "Hann"
        self.filter_type = "none"
        self.cutoff_freq = 1000

        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.file_tab = QWidget()
        self.tabs.addTab(self.file_tab, "File Analyzer")
        self.setup_file_tab()

    def setup_file_tab(self):
        layout = QVBoxLayout()
        self.file_tab.setLayout(layout)

        controls_layout = QHBoxLayout()

        load_btn = QPushButton("Load Audio File")
        load_btn.clicked.connect(self.load_audio_file)
        controls_layout.addWidget(load_btn)

        controls_layout.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["Hann", "Hamming", "Blackman", "None"])
        self.window_combo.currentTextChanged.connect(self.on_window_change)
        controls_layout.addWidget(self.window_combo)

        controls_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["none", "low", "high"])
        self.filter_combo.currentTextChanged.connect(self.on_filter_change)
        controls_layout.addWidget(self.filter_combo)

        controls_layout.addWidget(QLabel("Cutoff (Hz):"))
        self.cutoff_slider = QSlider(Qt.Horizontal)
        self.cutoff_slider.setMinimum(100)
        self.cutoff_slider.setMaximum(5000)
        self.cutoff_slider.setValue(1000)
        self.cutoff_slider.valueChanged.connect(self.on_cutoff_change)
        controls_layout.addWidget(self.cutoff_slider)

        self.cutoff_label = QLabel("1000 Hz")
        controls_layout.addWidget(self.cutoff_label)

        play_original_btn = QPushButton("Play Original")
        play_original_btn.clicked.connect(self.play_original)
        controls_layout.addWidget(play_original_btn)

        play_filtered_btn = QPushButton("Play Filtered")
        play_filtered_btn.clicked.connect(self.play_filtered)
        controls_layout.addWidget(play_filtered_btn)

        layout.addLayout(controls_layout)

        graphs_layout = QVBoxLayout()

        self.time_plot_before = pg.PlotWidget(title="Time Domain (Before Filter)")
        self.time_plot_after = pg.PlotWidget(title="Time Domain (After Filter)")
        graphs_layout.addWidget(self.time_plot_before)
        graphs_layout.addWidget(self.time_plot_after)

        self.fft_plot = pg.PlotWidget(title="Frequency Domain (FFT Before & After Filter)")
        graphs_layout.addWidget(self.fft_plot)

        self.spectrogram_plot = pg.ImageView()
        graphs_layout.addWidget(self.spectrogram_plot)

        layout.addLayout(graphs_layout)

    def load_audio_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose Audio File", "", "Audio Files (*.wav *.flac *.ogg *.mp3)")
        if path:
            data, sr = sf.read(path)
            if len(data.shape) > 1:
                data = data[:, 0]
            self.signal = data.astype(np.float32)
            self.sr = sr
            self.process_signal()

    def on_window_change(self, text):
        self.window_type = text
        self.process_signal()

    def on_filter_change(self, text):
        self.filter_type = text
        self.process_signal()

    def on_cutoff_change(self, value):
        self.cutoff_freq = value
        self.cutoff_label.setText(f"{value} Hz")
        self.process_signal()

    def play_original(self):
        if self.signal is not None:
            sd.play(self.signal, self.sr)

    def play_filtered(self):
        if self.filtered is not None:
            sd.play(self.filtered, self.sr)

    def process_signal(self):
        if self.signal is None:
            return

        windowed, window = apply_window_fixed(self.signal, self.window_type)
        filtered = windowed.copy()

        if self.filter_type == "low":
            filtered = butter_filter(windowed, self.sr, self.cutoff_freq, "lowpass")
        elif self.filter_type == "high":
            filtered = butter_filter(windowed, self.sr, self.cutoff_freq, "highpass")

        self.filtered = filtered

        self.update_time_plots(self.signal, filtered)
        self.update_fft_plots(self.signal, filtered, window)
        self.update_spectrogram(filtered)

    def update_time_plots(self, original, filtered):
        self.time_plot_before.clear()
        self.time_plot_after.clear()
        t = np.arange(len(original)) / self.sr
        self.time_plot_before.plot(t, original)
        self.time_plot_after.plot(t, filtered)

    def update_fft_plots(self, original, filtered, window):
        self.fft_plot.clear()
        xf1, mag1 = compute_fft(original, self.sr, window)
        xf2, mag2 = compute_fft(filtered, self.sr, window)
        self.fft_plot.plot(xf1, mag1, pen='y')
        self.fft_plot.plot(xf2, mag2, pen='c')
        self.fft_plot.setXRange(0, 5000)

    def update_spectrogram(self, filtered):
        f, t, Sxx = compute_spectrogram(filtered, self.sr)
        self.spectrogram_plot.setImage(Sxx, xvals=t)
        self.spectrogram_plot.setPredefinedGradient("inferno")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FourierAnalyzer()
    win.show()
    sys.exit(app.exec_())
