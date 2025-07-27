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
folder_path_20_25 = '/home/itk/Documents/AnneMarthe/Master/inSitu/2009/stn25'
folder_path_20_4 = '/home/itk/Documents/AnneMarthe/Master/inSitu/2009/stn4'
folder_path_17_25 = '/home/itk/Documents/AnneMarthe/Master/inSitu/1709/stn25'
folder_path_17_4 = '/home/itk/Documents/AnneMarthe/Master/inSitu/1709/stn4'

all_data_20_25, all_times_20_25 = read_dat_rrs_files(folder_path_20_25)
all_data_20_4, all_times_20_4 = read_dat_rrs_files(folder_path_20_4)
all_data_17_25, all_times_17_25 = read_dat_rrs_files(folder_path_17_25)
all_data_17_4, all_times_17_4 = read_dat_rrs_files(folder_path_17_4)

# === PROCESS FOR PLOTTING ===
# Assume wavelengths are the same for all files (take from first file)
wavelengths_20_25 = [w for (w, _) in all_data_20_25[0]]
wavelengths_20_4 = [w for (w, _) in all_data_20_4[0]]
wavelengths_17_25 = [w for (w, _) in all_data_17_25[0]]
wavelengths_17_4 = [w for (w, _) in all_data_17_4[0]]

# Create 2D reflectance matrix: shape (num_files, num_wavelengths)
reflectance_matrix_20_25 = np.array([
    [rrs for (_, rrs) in spectrum]
    for spectrum in all_data_20_25
])

reflectance_matrix_20_4 = np.array([
    [rrs for (_, rrs) in spectrum]
    for spectrum in all_data_20_4
])

reflectance_matrix_17_25 = np.array([
    [rrs for (_, rrs) in spectrum]
    for spectrum in all_data_17_25
])

reflectance_matrix_17_4 = np.array([
    [rrs for (_, rrs) in spectrum]
    for spectrum in all_data_17_4
])

# === PLOTTING ===
plt.figure(figsize=(10, 6))

# Plot each spectrum in light gray
#for i in range(reflectance_matrix.shape[0]):
#    plt.plot(wavelengths, reflectance_matrix[i], color='gray', linewidth=1, alpha=0.7)

# Plot median spectrum in red
median_spectrum_20_25 = np.median(reflectance_matrix_20_25, axis=0)
median_spectrum_20_4 = np.median(reflectance_matrix_20_4, axis=0)
median_spectrum_17_25 = np.median(reflectance_matrix_17_25, axis=0)
median_spectrum_17_4 = np.median(reflectance_matrix_17_4, axis=0)

plt.plot(wavelengths_17_25, median_spectrum_17_25, color='#e6194b', linewidth=2, label='Median Spectrum 17.09 st 2.5')
plt.plot(wavelengths_17_25, median_spectrum_17_4, color='#3cb44b', linewidth=2, label='Median Spectrum 17.09 st 4')
plt.plot(wavelengths_20_25, median_spectrum_20_25, color='#911eb4', linewidth=2, label='Median Spectrum 20.09 st 2.5')
plt.plot(wavelengths_20_4, median_spectrum_20_4, color='#0082c8', linewidth=2, label='Median Spectrum 20.09 st 4')


plt.xlabel("Wavelength [nm]")
plt.ylabel("RRS [sr$^{-1}$] ")
plt.title("RRS Median Spectra the 17.09 & 20.09 from st 2.5 & 4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("InSitu_median_20_17_24.pdf", dpi=300)
#plt.show()
