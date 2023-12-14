# %%
string = "A2b&$eL9HwCf7*1gXpQ3z"

# %%
test_str = string

# print("The original string is : " + str(test_str))

# using join() + ord() + format()
# Converting String to binary
binary_string = ''.join(format(ord(i), '08b') for i in test_str)

# printing result 
print("binary string: " + str(binary_string))



# %%
import numpy as np
import matplotlib.pyplot as plt

def binary_to_4pam(binary_string):
    if len(binary_string) % 2 != 0:
        raise ValueError("Binary string length must be even for 4-PAM modulation.")

    pam_mapping = {'00': -3, '01': -1, '10': 1, '11': 3}
    pam_symbols = [pam_mapping[binary_string[i:i+2]] for i in range(0, len(binary_string), 2)]
    return pam_symbols

# Example binary string
pam_symbols = binary_to_4pam(binary_string)

# Print binary string and 4-PAM symbols
print("Binary String:", binary_string)
print("4-PAM Symbols:", pam_symbols)

# Generate a continuous waveform by repeating each symbol for a short duration
t_continuous = np.linspace(0, len(pam_symbols), len(pam_symbols)*100)
pam_waveform = np.repeat(pam_symbols, 100)

# Plotting the 4-PAM symbols
plt.plot(t_continuous, pam_waveform, drawstyle='steps-pre')
plt.title('Transmit Waveform for Square Pulse')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()




# -------------------------------------------------------
# Assuming you have the binary string and pam_symbols
rrc_pulse = np.sinc(np.arange(0, len(pam_symbols)*100, 1) / 8)  # Assuming sps=8

# Convolve binary symbols with RRC pulse shaping filter
rrc_waveform = np.convolve(pam_symbols, rrc_pulse, mode='full')[:len(pam_symbols)*100]

# Plot the transmit waveform for RRC pulse
plt.plot(t_continuous, rrc_waveform, drawstyle='steps-pre')
plt.title('Transmit Waveform for RRC Pulse (Beta=0.25)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()



# -------------------------------------------------------------------

# Define matched filter for square pulse
matched_filter_square = np.ones(100)

# Define matched filter for RRC pulse
matched_filter_rrc = rrc_pulse[::-1][:len(pam_symbols)*100]


# ----------------------------------------------------------------
