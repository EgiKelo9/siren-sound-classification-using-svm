import numpy as np
import scipy.fftpack
from tqdm import tqdm
from scipy.io import wavfile
from cvxopt import matrix, solvers

def load_audio(filepath):
    sr, data = wavfile.read(filepath)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    return sr, data

def get_duration(signal, sr=44100):
    if isinstance(signal, str):
        sr, signal = load_audio(signal)
    elif isinstance(signal, np.ndarray):
        if signal.dtype != np.float32:
            signal = signal.astype(np.float32)
    else:
        raise ValueError("Signal must be a file path or a numpy array.")
    duration = len(signal) / sr
    return duration

def time_domain_features(signal):
    zcr = np.mean(np.abs(np.diff(np.sign(signal)))) / 2
    rms = np.sqrt(np.mean(signal**2))
    return zcr, rms

def frequency_domain_features(signal, sr):
    N = len(signal)
    fft = np.fft.rfft(signal)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(N, d=1/sr)

    spec_centroid = np.sum(freqs * mag) / np.sum(mag)
    spec_bandwidth = np.sqrt(np.sum(((freqs - spec_centroid)**2) * mag) / np.sum(mag))
    rolloff_thresh = 0.85 * np.sum(mag)
    rolloff_freq = freqs[np.where(np.cumsum(mag) >= rolloff_thresh)[0][0]]

    return spec_centroid, spec_bandwidth, rolloff_freq

def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700.0)
def mel_to_hz(mel): return 700 * (10**(mel / 2595.0) - 1)

def mfcc(signal, sr, n_mfcc=13, n_filters=26, n_fft=512, verbose=False):
    # 1. Framing
    frame_len = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    signal_len = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_len - frame_len)) / frame_step)) + 1

    pad_len = num_frames * frame_step + frame_len
    z = np.zeros((pad_len - signal_len))
    pad_signal = np.append(signal, z)

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # 2. Windowing
    frames *= np.hamming(frame_len)

    # 3. FFT and power spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))

    # 4. Mel filterbank
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_filters, int(n_fft / 2 + 1)))
    for m in range(1, n_filters + 1):
        f_m_minus, f_m, f_m_plus = bin[m - 1], bin[m], bin[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_fbanks = np.log(filter_banks)

    # 5. DCT
    mfccs = scipy.fftpack.dct(log_fbanks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    if verbose: 
        return mfccs
    return np.mean(mfccs, axis=0)

def extract_all_features(sr, signal):
    zcr, rms = time_domain_features(signal)
    centroid, bandwidth, rolloff = frequency_domain_features(signal, sr)
    mfccs = mfcc(signal, sr)
    return np.array([zcr, rms, centroid, bandwidth, rolloff] + mfccs.tolist())

def uniform_data(extracted_features):
    processed_features = []
    for feature in extracted_features:
        if isinstance(feature, np.ndarray):
            processed_features.append(feature.item())
        else:
            processed_features.append(feature)
    return np.array(processed_features).reshape(1, -1)


class SVM:
    
    # Fungsi konstruktor untuk menginisialisasi parameter SVM
    def __init__(self, kernel='rbf', C=10.0, degree=3, gamma=0.01):
        self.kernel_type = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.models = {}
        self.is_multiclass = False
    
    # Fungsi untuk mendapatkan parameter SVM
    def get_params(self):
        return {
            "C": self.C,
            "kernel": self._compute_kernel,
            "gamma": self.gamma,
            "degree": self.degree,
        }

    # Fungsi untuk mengatur parameter SVM
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    # Fungsi untuk menghitung kernel
    def _compute_kernel(self, x, y):
        if self.kernel_type == 'linear':
            return np.dot(x, y)
        elif self.kernel_type == 'poly':
            return (1 + self.gamma * np.dot(x, y)) ** self.degree
        elif self.kernel_type == 'rbf':
            return np.exp(-self._gamma * np.linalg.norm(x - y) ** 2)
        else:
            raise ValueError("Unknown kernel")

    # Fungsi untuk melakukan proses fitting model SVM
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.is_multiclass = n_classes > 2

        if self.is_multiclass:
            self.models = {}
            print(f"Training {n_classes} OvR SVM models...\n")
            for cls in tqdm(self.classes, desc="OvR SVM Training"):
                y_binary = np.where(y == cls, 1, -1)
                model = SVM(kernel=self.kernel_type, C=self.C, degree=self.degree, gamma=self.gamma)
                model.fit(X, y_binary)
                self.models[cls] = model
        else:
            y = y.astype(float)
            n_samples, n_features = X.shape
            self.X = X
            self.y = y

            if self.kernel_type == 'rbf':
                self._gamma = self.gamma if self.gamma else 1 / n_features

            # Gram matrix
            K = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = self._compute_kernel(X[i], X[j])

            P = matrix(np.outer(y, y) * K)
            q = matrix(-np.ones(n_samples))
            A = matrix(y.reshape(1, -1))
            b = matrix(0.0)

            if self.C is None:
                G = matrix(-np.eye(n_samples))
                h = matrix(np.zeros(n_samples))
            else:
                G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
                h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)
            alphas = np.ravel(solution['x'])

            # Support vectors
            sv = alphas > 1e-5
            self.alphas = alphas[sv]
            self.support_vectors = X[sv]
            self.support_vector_labels = y[sv]
            print(f"Self Alphas: {self.alphas}")
            print(f"Alphas shape: {self.alphas.shape}")
            print(f"Support vectors shape: {self.support_vectors.shape}")

            # Intercept
            self.b = np.mean([
                y_k - np.sum(self.alphas * self.support_vector_labels *
                             [self._compute_kernel(x_k, x_i) for x_i in self.support_vectors])
                for (x_k, y_k) in zip(self.support_vectors, self.support_vector_labels)
            ])

    # Fungsi untuk menghitung nilai keputusan
    def project(self, X):
        if self.is_multiclass:
            decision_values = np.column_stack([
                model.project(X) for model in self.models.values()
            ])
            return decision_values
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for alpha, sv_y, sv in zip(self.alphas, self.support_vector_labels, self.support_vectors):
                    s += alpha * sv_y * self._compute_kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b
    
    # Fungsi untuk menghitung probabilitas prediksi
    def predict_proba(self, X):
        if not self.is_multiclass:
            raise NotImplementedError("predict_proba hanya didukung untuk mode multiclass OvR")

        decision = self.project(X)
        # Softmax untuk setiap baris
        exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))  # stabilisasi
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    # Fungsi untuk melakukan prediksi
    def predict(self, X):
        if self.is_multiclass:
            decision = self.project(X)
            predictions = np.argmax(decision, axis=1)
            return self.classes[predictions]
        else:
            return np.sign(self.project(X))