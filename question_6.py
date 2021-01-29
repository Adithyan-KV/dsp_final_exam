import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

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

    EAHN_1 = e(test_signal, 1)
    EAHN_2 = e(test_signal, 1)
    EAHN_3 = e(test_signal, 1)
    EAHN_4 = e(test_signal, 1)

    # Plotting EAHN
    fig, plots = plt.subplots(2,2)
    fig.suptitle("4 realizations of EAHN")
    plots[0,0].plot(EAHN_1)
    plots[0,1].plot(EAHN_2)
    plots[1,0].plot(EAHN_3)
    plots[1,1].plot(EAHN_4)
    plt.show()

    #---------------------------------------------------
    # Question b
    #---------------------------------------------------

    A_values = np.random.uniform(0,1,length)
    # setting A_1=1
    A_values[0]=1
    UAHN_1 = e(test_signal, A_values)
    UAHN_2 = e(test_signal, A_values)
    UAHN_3 = e(test_signal, A_values)
    UAHN_4 = e(test_signal, A_values)

    # Plotting UAHN
    fig, plots = plt.subplots(2,2)
    fig.suptitle("4 realizations of UAHN")
    plots[0,0].plot(UAHN_1)
    plots[0,1].plot(UAHN_2)
    plots[1,0].plot(UAHN_3)
    plots[1,1].plot(UAHN_4)
    plt.show()

    #---------------------------------------------------
    # Question c
    #---------------------------------------------------
    A_values = np.random.uniform(0,1,len(speech_signal))
    A_values[0]=1
    EAHN_1 = e(speech_signal, A_values)
    EAHN_2 = e(speech_signal, A_values)
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
    A_values = np.ones(len(speech_signal))
    UAHN_1 = e(speech_signal, A_values)
    UAHN_2 = e(speech_signal, A_values)
    db = [0,5,10,15]
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


def e(n, Ak, f0=600):
    length = len(n)
    k = np.arange(0,length,1)
    phi_k = np.random.uniform(0,2*np.pi,length)
    series = Ak*np.sin(2*np.pi*k*n+phi_k)
    en = np.cumsum(series)
    return en

def get_constant_for_snr(signal, noise, snr):
    Es = np.sum(np.square(signal))
    En = np.sum(np.square(noise))
    constant = (Es/En)*10**(-snr/10)
    return constant

def add_noise_to_signal(signal, noise, snr):
    k = get_constant_for_snr(signal, noise, snr)
    noisy_signal = signal + k*noise
    return noisy_signal

if __name__ == '__main__':
    main()