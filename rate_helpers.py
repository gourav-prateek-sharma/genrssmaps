def compute_rate_from_arr(rss_array, bandwidth_hz=1e6, noise_dbm=-100, snr_gap_db=0):
    """
    Compute average achievable rate from RSS array (in dBm).
    Args:
        rss_array (np.ndarray): RSS values in dBm
        bandwidth_hz (float): Channel bandwidth in Hz
        noise_dbm (float): Noise power in dBm
        snr_gap_db (float): SNR gap in dB (for non-ideal coding)
    Returns:
        float: Average rate in bits/s/Hz
    """
    import numpy as np
    # Convert RSS and noise to linear scale
    snr_linear = 10 ** ((rss_array - noise_dbm - snr_gap_db) / 10)
    rate = np.log2(1 + snr_linear)
    avg_rate = np.mean(rate)
    return avg_rate

def compute_rate_from_csv(csv_file, bandwidth_hz=1e6, noise_dbm=-100, snr_gap_db=0):
    import pandas as pd
    rss_array = pd.read_csv(csv_file, header=None).values
    return compute_rate_from_arr(rss_array, bandwidth_hz, noise_dbm, snr_gap_db)
