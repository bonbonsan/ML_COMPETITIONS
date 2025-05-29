import numpy as np
from scipy.signal import convolve, hann, hilbert
from sklearn.linear_model import LinearRegression


def compute_fft_features(signal: np.ndarray, prefix: str = "fft") -> dict:
    """
    Compute FFT-based statistical features (real & imag parts).

    Args:
        signal: 1D numpy array
        prefix: Prefix for feature names

    Returns:
        Dictionary of FFT features
    """
    fft = np.fft.fft(signal)
    real = np.real(fft)
    imag = np.imag(fft)

    features = {
        f"{prefix}_real_mean": real.mean(),
        f"{prefix}_real_std": real.std(),
        f"{prefix}_real_min": real.min(),
        f"{prefix}_real_max": real.max(),
        f"{prefix}_imag_mean": imag.mean(),
        f"{prefix}_imag_std": imag.std(),
        f"{prefix}_imag_min": imag.min(),
        f"{prefix}_imag_max": imag.max(),
    }
    return features


def compute_trend_feature(arr: np.ndarray, abs_values: bool = False) -> float:
    """
    Calculate linear trend (slope) of the given 1D array.

    Args:
        arr: 1D array
        abs_values: If True, take abs(arr) before fitting

    Returns:
        Slope (trend)
    """
    idx = np.arange(len(arr)).reshape(-1, 1)
    y = np.abs(arr) if abs_values else arr
    lr = LinearRegression()
    lr.fit(idx, y)
    return lr.coef_[0]


def compute_sta_lta(
    x: np.ndarray,
    length_sta: int,
    length_lta: int
) -> np.ndarray:
    """
    Compute classic Short-Term Average / Long-Term Average (STA/LTA) ratio.

    Args:
        x: Input signal
        length_sta: Window size for short-term average
        length_lta: Window size for long-term average

    Returns:
        STA/LTA ratio array
    """
    x2 = x ** 2
    sta = np.cumsum(x2, dtype=float)
    lta = sta.copy()

    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta

    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    sta[:length_lta - 1] = 0
    lta[lta < np.finfo(0.0).tiny] = np.finfo(0.0).tiny

    return sta / lta


def compute_hilbert_envelope_mean(signal: np.ndarray) -> float:
    """
    Compute the mean amplitude envelope using Hilbert transform.

    Args:
        signal: 1D array

    Returns:
        Mean of the envelope
    """
    return np.abs(hilbert(signal)).mean()


def compute_hann_window_mean(signal: np.ndarray, window_size: int = 150) -> float:
    """
    Apply Hann window smoothing and return mean of result.

    Args:
        signal: 1D array
        window_size: Size of the Hann window

    Returns:
        Mean of smoothed signal
    """
    window = hann(window_size)
    smoothed = convolve(signal, window, mode="same") / window.sum()
    return smoothed.mean()
