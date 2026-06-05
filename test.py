from MRKLT import MRKLT
import numpy as np
import matplotlib.pyplot as plt

# dimensions:
M = 1e4 # realizations
N = 1e3 # samples per realization

# sinewave signal generation:
# uniform distribution of phase
def generate_signal(M, N, f0, fs):
    t = np.arange(N) / fs
    signal = np.zeros((int(M), int(N)))
    for i in range(int(M)):
        phi = np.random.uniform(0, 2 * np.pi)
        signal[i, :] = np.sin(2 * np.pi * f0 * t + phi)
    return signal

# noise generation:
# complex coloured noise using white noise convolved with a Hanning window
def generate_noise(M, N):
    white_noise = np.random.normal(0, 1, (int(M), int(N)))
    hanning_window = np.hanning(int(N))
    noise = np.array([np.convolve(white_noise[i, :], hanning_window, mode='same') for i in range(int(M))])
    return noise

# plot eigenspectrum for a specified realization:
def plot_eigenspectrum(eigenvalues, n_eigenvalues_of_interest=1):
    fig, ax = plt.subplots()
    ax.scatter(np.arange(n_eigenvalues_of_interest, len(eigenvalues)), eigenvalues[n_eigenvalues_of_interest:], color='black', 
               marker='.',label='eigenvalues rejected')
    ax.scatter(np.arange(n_eigenvalues_of_interest), eigenvalues[:n_eigenvalues_of_interest], color='red', 
               marker='.', label=f'eigenvalues considered: {n_eigenvalues_of_interest}')
    ax.set(
        xlabel='eigenvalue index (sorted in descending order)',
        ylabel='eigenvalue magnitude',
    )
    ax.legend()
    ax.grid()
    return fig, ax

# plot psd comparison for a specified realization:
def plot_psd_comparison(signal, noise, reconstructed_signal, fs, realization_index=0):
    fig, ax = plt.subplots(2, 1, sharex=True)
    received_signal = signal + noise
    ax[0].psd(received_signal[realization_index, :], NFFT=1024, Fs=fs, label='received signal', color='blue')
    ax[1].psd(reconstructed_signal[realization_index, :], NFFT=1024, Fs=fs, label='reconstructed signal', color='red')
    ax[1].psd(signal[realization_index, :], NFFT=1024, Fs=fs, label='original signal', color='green')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('PSD (dB/Hz)')
    ax[1].set_ylabel('PSD (dB/Hz)')
    ax[0].legend()
    ax[1].legend()
    ax[0].grid()
    ax[1].grid()
    return fig, ax

#=====================================================================================================================#

def main():
    # parameters:
    fs = 1000 # sampling frequency in Hz
    f0 = 0.6*fs # normalized signal frequency

    # SNR values in dB:
    SNR_dB = -5
    SNR_linear = 10 ** (SNR_dB / 10)

    # generate signal and noise:
    signal = generate_signal(M, N, f0, fs)
    noise = generate_noise(M, N)
    # scale noise to achieve desired SNR:
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / SNR_linear
    scaled_noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))

    # initialize MRKLT:
    mrklt = MRKLT()
    mrklt.load_signal(signal)
    mrklt.load_noise(scaled_noise)

    n_eigenvectors_of_interest = 1

    # run MRKLT:
    reconstructed_signal =mrklt.reconstruct_signal(n_eigenvectors=n_eigenvectors_of_interest)
    eigenvalues = mrklt.get_eigenvalues()

    fig1, ax1 =plot_eigenspectrum(eigenvalues, n_eigenvectors_of_interest)

    fig2, ax2 =plot_psd_comparison(signal, scaled_noise, reconstructed_signal, fs, realization_index=3000)

    plt.show()

if __name__ == "__main__":
    main()