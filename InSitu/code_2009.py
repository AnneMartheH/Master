import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

def read_dat_rrs_files(folder_path):
    """
    Reads Rrs data from .dat files in a given folder, extracting wavelength and reflectance values.

    Returns:
    - all_data: List of lists, each sublist is a list of (wavelength, reflectance) tuples per file
    - all_times: List of time strings extracted from filenames (HH:MM:SS)
    """
    all_data = []
    all_times = []

    dat_files = sorted(glob.glob(f'{folder_path}/*.dat'))
    print(dat_files)
    for file_path in dat_files:
        filename = os.path.basename(file_path)
        match = re.search(r'(\d{6})\.dat$', filename)
        if match:
            time_str = match.group(1)
            time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
            all_times.append(time)
        else:
            print(f"Time not found in the filename: {filename}")
            continue

        file_data = []
        with open(file_path, 'r') as file:
            for line in file:
                if len(line.strip()) == 0:
                    continue
                columns = line.split()
                if len(columns) == 2:
                    try:
                        wavelength = float(columns[0])
                        reflectance = float(columns[1])
                        file_data.append((wavelength, reflectance))
                    except ValueError:
                        continue
        all_data.append(file_data)

    return all_data, all_times


# === USE FUNCTION ===
folder_path = '/home/itk/Documents/AnneMarthe/Master/inSitu/2009/stn25'
all_data, all_times = read_dat_rrs_files(folder_path)

# === PROCESS FOR PLOTTING ===
# Assume wavelengths are the same for all files (take from first file)
wavelengths = [w for (w, _) in all_data[0]]

# Create 2D reflectance matrix: shape (num_files, num_wavelengths)
reflectance_matrix = np.array([
    [rrs for (_, rrs) in spectrum]
    for spectrum in all_data
])

# === PLOTTING ===
plt.figure(figsize=(10, 6))

# Plot each spectrum in light gray
for i in range(reflectance_matrix.shape[0]):
    plt.plot(wavelengths, reflectance_matrix[i], color='gray', linewidth=1, alpha=0.7)

# Plot median spectrum in red
median_spectrum = np.median(reflectance_matrix, axis=0)
plt.plot(wavelengths, median_spectrum, color='red', linewidth=2, label='Median Spectrum')

plt.xlabel("Wavelength [nm]")
plt.ylabel("RRS [sr$^{-1}$] ")
plt.title("RRS Spectra the 20.09.2024 with Median from Station 2.5")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("InSitu_2009_24.pdf", dpi=300)
#plt.show()
