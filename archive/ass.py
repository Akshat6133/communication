import numpy as np
# from scipy import signal
import matplotlib.pyplot as plt


def rrcf(N, beta, Ts, Fs):

    T_delta = 1.0 / Fs
    time_index = np.arange(-N / 2, N / 2) * T_delta
    # sample_num = np.arange(-N / 2, N / 2)

    h_rrc = np.zeros(len(time_index), dtype=float)

    for idx, t in enumerate(time_index):
        if t == 0.0:
            h_rrc[idx] = 1.0 - beta + (4 * beta / np.pi)
        elif beta != 0 and t == Ts / (4 * beta):
            h_rrc[idx] = (beta / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * (np.sin(np.pi / (4 * beta))))
                + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * beta))))
            )
        elif beta != 0 and t == -Ts / (4 * beta):
            h_rrc[idx] = (beta / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * (np.sin(np.pi / (4 * beta))))
                + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * beta))))
            )
        else:
            h_rrc[idx] = (
                np.sin(np.pi * t * (1 - beta) / Ts)
                + 4 * beta * (t / Ts) * np.cos(np.pi * t * (1 + beta) / Ts)
            ) / (np.pi * t * (1 - (4 * beta * t / Ts) ** 2) / Ts)

    return time_index, h_rrc


num_symbols = 20
sps = 8

bits = np.random.randint(
    0, 2, num_symbols * 2
)  # Our data to be transmitted, 1's and 0's
print(bits)


mapping = {
    "00": -3,
    "01": -1,
    "11": 1,
    "10": 3,
}
y = np.array([])
for bit in range(0, len(bits), 2):
    bit_mapping = str(bits[bit]) + str(bits[bit + 1])
    pulse = np.zeros(sps)
    pulse[0] = mapping[bit_mapping]
    y = np.concatenate((y, pulse))


plt.figure(0)
plt.plot(y, ".-")
plt.grid(True)
plt.show()


# Test the function
N = 101
beta = 0.25
Ts = 0.8
Fs = 10

time_index, h_rrc = rrcf(N, beta, Ts, Fs)

# Plot the filter
plt.figure()
plt.plot(time_index, h_rrc)
plt.title("Root Raised Cosine Filter")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

y_shaped = np.convolve(y, h_rrc)
plt.plot(y_shaped, ".-")
plt.show()


# # Assuming yc and Fs are defined elsewhere in your code
# # yc is your signal, Fs is the sampling frequency

X = np.fft.fftshift(np.fft.fft(y_shaped, 800))
f_axis = np.arange(0, 800) * Fs / 800 - Fs / 2

plt.plot(f_axis / 1e6, 20 * np.log10(np.abs(X)))
plt.title("Spectrum of modulated signal")
plt.xlabel("f in MHz")
plt.ylabel("PSD in dB")
plt.show()


taps = 8
x_sq = np.arange(taps)
pulse = np.ones_like(x_sq)

# Generate the square pulse
# pulse = np.ones(N)

# Plot the square pulse
plt.figure()
plt.stem(pulse, use_line_collection=True)
plt.title("Square Pulse")
plt.xlabel("Tap")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


x_shq = np.convolve(y, pulse)
plt.plot(x_shq, ".-")
plt.show()


X = np.fft.fftshift(np.fft.fft(x_shq, 800))
f_axis = np.arange(0, 800) * Fs / 800 - Fs / 2

plt.plot(f_axis / 1e6, 20 * np.log10(np.abs(X)))
plt.title("Spectrum of modulated signal")
plt.xlabel("f in MHz")
plt.ylabel("PSD in dB")
plt.show()


rec_conv = np.convolve(x_shq, pulse) / 8

# Taking every 7th asmple
evey_seven_sample = []
length = []
count = 1
for i in range(7, len(rec_conv), 8):
    length.append(count)
    count += 1
    evey_seven_sample.append(rec_conv[i])
    # print(rec_conv[i], end=" ")

plt.scatter(length, evey_seven_sample)
plt.grid(True)
plt.show()
plt.plot(rec_conv, ".-")
plt.show()