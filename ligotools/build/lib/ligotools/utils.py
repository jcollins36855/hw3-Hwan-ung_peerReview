import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.interpolate import interp1d
from scipy.signal import filtfilt

def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0, 2048, Nt // 2 + 1)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht

def write_wavfile(filename, fs,data):
    """function to keep the data within integer limits, and write to wavfile:
    """
    d = np.int16(data/np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write('audio/' + filename, int(fs), d)

def reqshift(data,fshift=100,sample_rate=4096):
    """
    Frequency shift the signal by constant
    function that shifts frequency of a band-passed signal
    """
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real,nbins) + 1j*np.roll(x.imag,nbins)
    y[0:nbins]=0.
    z = np.fft.irfft(y)
    return z

def plot_matched_filter_analysis(
    dets, dt, strain_L1, strain_H1,
    datafreq, template_fft,
    fs, NFFT, psd_window, NOVL,
    dwindow, df, time, template,
    ab, bb, normalization,
    strain_H1_whitenbp, strain_L1_whitenbp,
    tevent, eventname, plottype="png",
    figures_dir="figures", make_plots=True
):
    for det in dets:
        if det == 'L1': data = strain_L1.copy()
        else:           data = strain_H1.copy()

        data_psd, freqs = mlab.psd(data, Fs = fs, NFFT = NFFT, window=psd_window, noverlap=NOVL)
        data_fft = np.fft.fft(data*dwindow) / fs
        power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
        optimal = data_fft * template_fft.conjugate() / power_vec
        optimal_time = 2*np.fft.ifft(optimal)*fs

        sigmasq = 1*(template_fft * template_fft.conjugate() / power_vec).sum() * df
        sigma = np.sqrt(np.abs(sigmasq))
        SNR_complex = optimal_time/sigma

        peaksample = int(data.size / 2)
        SNR_complex = np.roll(SNR_complex,peaksample)
        SNR = abs(SNR_complex)

        indmax = np.argmax(SNR)
        timemax = time[indmax]
        SNRmax = SNR[indmax]

        d_eff = sigma / SNRmax
        horizon = sigma/8

        phase = np.angle(SNR_complex[indmax])
        offset = (indmax-peaksample)

        template_phaseshifted = np.real(template*np.exp(1j*phase))
        template_rolled = np.roll(template_phaseshifted,offset) / d_eff
    
        template_whitened = whiten(template_rolled,interp1d(freqs, data_psd),dt)
        template_match = filtfilt(bb, ab, template_whitened) / normalization
    
        print('For detector {0}, maximum at {1:.4f} with SNR = {2:.1f}, D_eff = {3:.2f}, horizon = {4:0.1f} Mpc' 
              .format(det,timemax,SNRmax,d_eff,horizon))

        if make_plots:
            if det == 'L1': 
                pcolor='g'
                strain_whitenbp = strain_L1_whitenbp
                template_L1 = template_match.copy()
            else:
                pcolor='r'
                strain_whitenbp = strain_H1_whitenbp
                template_H1 = template_match.copy()

            plt.figure(figsize=(10,8))
            plt.subplot(2,1,1)
            plt.plot(time-timemax, SNR, pcolor,label=det+' SNR(t)')
            plt.grid('on')
            plt.ylabel('SNR')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.legend(loc='upper left')
            plt.title(det+' matched filter SNR around event')

            plt.subplot(2,1,2)
            plt.plot(time-timemax, SNR, pcolor,label=det+' SNR(t)')
            plt.grid('on')
            plt.ylabel('SNR')
            plt.xlim([-0.15,0.05])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.legend(loc='upper left')
            plt.savefig(f"figures/{eventname}_{det}_SNR.{plottype}")

            plt.figure(figsize=(10,8))
            plt.subplot(2,1,1)
            plt.plot(time-tevent,strain_whitenbp,pcolor,label=det+' whitened h(t)')
            plt.plot(time-tevent,template_match,'k',label='Template(t)')
            plt.ylim([-10,10])
            plt.xlim([-0.15,0.05])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.ylabel('whitened strain (units of noise stdev)')
            plt.legend(loc='upper left')
            plt.title(det+' whitened data around event')

            plt.subplot(2,1,2)
            plt.plot(time-tevent,strain_whitenbp-template_match,pcolor,label=det+' resid')
            plt.ylim([-10,10])
            plt.xlim([-0.15,0.05])
            plt.grid('on')
            plt.xlabel('Time since {0:.4f}'.format(timemax))
            plt.ylabel('whitened strain (units of noise stdev)')
            plt.legend(loc='upper left')
            plt.title(det+' Residual whitened data after subtracting template around event')
            plt.savefig(f"figures/{eventname}_{det}_matchtime.{plottype}")
                 
            plt.figure(figsize=(10,6))
            template_f = np.absolute(template_fft)*np.sqrt(np.abs(datafreq)) / d_eff
            plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
            plt.loglog(freqs, np.sqrt(data_psd),pcolor, label=det+' ASD')
            plt.xlim(20, fs/2)
            plt.ylim(1e-24, 1e-20)
            plt.grid()
            plt.xlabel('frequency (Hz)')
            plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
            plt.legend(loc='upper left')
            plt.title(det+' ASD and template around event')
            plt.savefig(f"figures/{eventname}_{det}_matchfreq.{plottype}")
