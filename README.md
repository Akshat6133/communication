# communication

##part 1

1. Deadline on or before Thursday 23-11-2023 (No late submission allowed)
2. Generate a binary stream for 20 symbols for 4-PAM modulation
3. Bit rate of 20Mbps
4. sps 8 (Sample per symbol)
5. Plot the transmit waveform for square pulse
6. Plot the transmit waveform for RRC pulse with beta=0.25

7. Plot the spectrum of the transmitted signal using FFT function of matlab/Python

Hint for plotting spectrum: R=20 Mbps, which implies 10*10^6 symbols per second (2 bits per symbol for 4-PAM). Samples per symbol (sps) is 8. Thus the sampling rate is Fs=8*10*10^6.  Thus to plot FFT assume Fs=80MHz. Let vector xm is the samples of modulated signal, ie, 8*20=160 samples.


X=fftshift(fft(yc,800));
f_axis=(0:800-1)*Fs/800-Fs/2;
plot(f_axis/1e6,20*log10(abs(X)))
title('Spectrum of modulated signal')
xlabel('f in MHz')
ylabel('PSD in dB')
grid on

8. Choose the appropriate matched filter for 5 and 6
9. Assume perfect synchronization at the receiver and plot the constellation diagram for both 5&6 with matched filter mentioned in 8.

##part 2 

This is simple extension of part 1 of baseband communication 

1. Generate a sine wave of frequency fc=Fs/4, where Fs is the sampling frequency in the previous assignment. 
2. Multiply the carrier signal with the modulated signal at the transmitter after the pulse shaping filter. x_pb(n)=x_bb(n).*cos(2*pi*fc*n/Fs)
x_pb means passband signal, and x_bb is the baseband signal generated in previous assignment
3. Multiply the transmitted signal again with the carrier signal
     x_rx(n)=x_pb(n).*cos(2*pi*fc*n/Fs)
4. Then pass it through a LPF with cut-off frequency f_BW (bandwidth of x_mod)
Hint: use the spectrum to find the bandwidth of the modulated signal from previous assignment.
     from scipy import signal

     numtaps = 25
     f = f_BW/Fs
     h=signal.firwin(numtaps, f)
    yc=filter(x_rx,h)

5. Continue with the baseband receiver processing as in previous assignment.
