import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.fftpack as fft

def main():
    # all SI units
    sampling_rate = 8000
    time_period = 20e-3
    length = int(time_period*sampling_rate)
    test_signal = np.zeros(length)
    speech_signal = (wavfile.read('speech.wav'))[1]

    #---------------------------------------------------
    # Question a
    #---------------------------------------------------

    EAHN_1 = get_eahn(test_signal)
    EAHN_2 = get_eahn(test_signal)
    EAHN_3 = get_eahn(test_signal)
    EAHN_4 = get_eahn(test_signal)

    # Plotting EAHN
    fig, plots = plt.subplots(2,2)
    fig.suptitle("4 realizations of EAHN")
    plots[0,0].plot(EAHN_1)
    plots[0,0].set_xlabel('n')
    plots[0,0].set_ylabel('e[n]')
    plots[0,1].plot(EAHN_2)
    plots[0,1].set_xlabel('n')
    plots[0,1].set_ylabel('e[n]')
    plots[1,0].plot(EAHN_3)
    plots[1,0].set_xlabel('n')
    plots[1,0].set_ylabel('e[n]')
    plots[1,1].plot(EAHN_4)
    plots[1,1].set_xlabel('n')
    plots[1,1].set_ylabel('e[n]')
    plt.show()

    #---------------------------------------------------
    # Question b
    #---------------------------------------------------

    UAHN_1 = get_uahn(test_signal)
    UAHN_2 = get_uahn(test_signal)
    UAHN_3 = get_uahn(test_signal)
    UAHN_4 = get_uahn(test_signal)

    # Plotting UAHN
    fig, plots = plt.subplots(2,2)
    fig.suptitle("4 realizations of UAHN")
    plots[0,0].plot(UAHN_1)
    plots[0,0].set_xlabel('n')
    plots[0,0].set_ylabel('e[n]')
    plots[0,1].plot(UAHN_2)
    plots[0,1].set_xlabel('n')
    plots[0,1].set_ylabel('e[n]')
    plots[1,0].plot(UAHN_3)
    plots[1,0].set_xlabel('n')
    plots[1,0].set_ylabel('e[n]')
    plots[1,1].plot(UAHN_4)
    plots[1,1].set_xlabel('n')
    plots[1,1].set_ylabel('e[n]')
    plt.show()

    #---------------------------------------------------
    # Question c
    #---------------------------------------------------
    EAHN_1 = get_eahn(speech_signal)
    EAHN_2 = get_eahn(speech_signal)
    db = [0,5,10,15]
    fig, plots = plt.subplots(4,2)
    fig.suptitle("EAHN added at different SNRs")
    for index,snr in enumerate(db):
        noisy_1 = add_noise_to_signal(speech_signal, EAHN_1, snr)
        noisy_2 = add_noise_to_signal(speech_signal, EAHN_2, snr)
        plots[index,0].plot(noisy_1)
        plots[index,0].set_title(f'First realization of noisy signal with SNR:{snr}')
        plots[index,0].xaxis.set_visible(False)
        plots[index,1].plot(noisy_2)
        plots[index,1].set_title(f'Second realization of noisy signal with SNR:{snr}')
        plots[index,1].xaxis.set_visible(False)
    plt.show()

    #---------------------------------------------------
    # Question d
    #---------------------------------------------------
    UAHN_1 = get_uahn(speech_signal)
    UAHN_2 = get_uahn(speech_signal)
    db,m = [0,5,10,15],0.45
    fig, plots = plt.subplots(4,2)
    fig.suptitle("UAHN added at different SNRs")
    for index,snr in enumerate(db):
        noisy_1 = add_noise_to_signal(speech_signal, UAHN_1, snr)
        noisy_2 = add_noise_to_signal(speech_signal, UAHN_2, snr)
        plots[index,0].plot(noisy_1)
        plots[index,0].set_title(f'First realization of noisy signal with SNR:{snr}')
        plots[index,0].xaxis.set_visible(False)
        plots[index,1].plot(noisy_2)
        plots[index,1].set_title(f'Second realization of noisy signal with SNR:{snr}')
        plots[index,1].xaxis.set_visible(False)
    plt.show()

    #---------------------------------------------------
    # Question e
    #---------------------------------------------------
    snrs = [0,5,10,15]
    print('--------------------EAHN----------------------------')
    for snr in snrs:
        snr_list = []
        for _ in range(10):
            EAHN = get_eahn(speech_signal)
            noisy_signal = add_noise_to_signal(speech_signal, EAHN, snr)
            fft_signal = fft.fft(noisy_signal)
            power_spectrum = fft_signal*np.conj(fft_signal)
            power_noise = 0
            for harmonic_freq in range(600,4000,600):
                power_noise += np.sum(power_spectrum[harmonic_freq-2:harmonic_freq+2])
            power_signal = power_spectrum.sum()-2*power_noise
            snr_calc = 10*np.log10(np.abs(power_signal/power_noise))
            snr_list.append(m*snr_calc)
        snr_mean = np.mean(snr_list)
        snr_std = np.std(snr_list)
        print(f'For {snr}db noise, mean SNR:{snr_mean} with std:{snr_std}')

    #---------------------------------------------------
    # Question f
    #---------------------------------------------------
    snrs = [0,5,10,15]
    print('--------------------UAHN----------------------------')
    for snr in snrs:
        snr_list = []
        for _ in range(10):
            UAHN = get_uahn(speech_signal)
            noisy_signal = add_noise_to_signal(speech_signal, UAHN, snr)
            fft_signal = fft.fft(noisy_signal)
            power_spectrum = fft_signal*np.conj(fft_signal)
            power_noise = 0
            for harmonic_freq in range(600,4000,600):
                power_noise += np.sum(power_spectrum[harmonic_freq-2:harmonic_freq+2])
            power_signal = power_spectrum.sum()-2*power_noise
            snr_calc = 10*m*np.log10(np.abs(power_signal/power_noise))
            snr_list.append(snr_calc)
        snr_mean = np.mean(snr_list)
        snr_std = np.std(snr_list)
        print(f'For {snr}db noise, mean SNR:{snr_mean} with std:{snr_std}')

def get_eahn(signal,f0=600,Fs=8000):
    N = len(signal)
    K = 6
    Ak = 1
    e = np.zeros(N)
    phi = np.random.uniform(0,2*np.pi,K)
    for n in range(N):
        for k in range(K):
            e[n]+=Ak*np.sin(2*np.pi*k*f0/Fs*n+phi[k])
    return e

def get_uahn(signal,f0=600,Fs=8000):
    N = len(signal)
    K = 6
    Ak = np.random.uniform(0,1,K)
    Ak[0]=1
    e = np.zeros(N)
    phi = np.random.uniform(0,2*np.pi,K)
    for n in range(N):
        for k in range(K):
            e[n]+=Ak[k]*np.sin(2*np.pi*k*f0/Fs*n+phi[k])
    return e

def get_constant_for_snr(signal, noise, snr):
    Es = np.sum(np.square(signal))
    En = np.sum(np.square(noise))
    constant = (Es/En)*(10**(-snr/10))
    return constant

def add_noise_to_signal(signal, noise, snr):
    k = get_constant_for_snr(signal, noise, snr)
    noisy_signal = signal + k*noise
    return noisy_signal

if __name__ == '__main__':
    main()