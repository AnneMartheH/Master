import numpy as np 
from global_land_mask import globe 
import math 
import cv2
import matplotlib.pyplot as plt

def lat_long_to_piksel(target_lat, target_lon, latitudes, longitudes):  
    piksel_row = []
    piksel_col = []

    for i in range(4):
        lat_diff = np.abs(latitudes - target_lat[i])
        lon_diff = np.abs(longitudes - target_lon[i])

        row, col = np.unravel_index(np.argmin(lat_diff + lon_diff), latitudes.shape)

        piksel_row.append(row)
        piksel_col.append(col)

    return piksel_row, piksel_col


def piksel_to_area(piksel_row, piksel_col, datacube): ## change name to pixel_to_area()
    image = datacube[:,:,100]
    polygon_points = np.array([
        [piksel_col[0], piksel_row[0]],  # Top-left
        [piksel_col[1], piksel_row[1]],  # Top-right
        [piksel_col[2], piksel_row[2]],  # Bottom-right
        [piksel_col[3], piksel_row[3]]   # Bottom-left
    ], dtype=np.int32)

    image_np = image.values
    mask = np.zeros(image_np.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 1)

    row, col = np.where(mask == 1)
    return row, col


def area_within_watermask(rows, cols, latitudes, longitudes):
    rows_water = []
    cols_water = []

    for row, col in zip(rows, cols):
        if globe.is_ocean(latitudes[row][col], longitudes[row][col]):
            rows_water.append(row)
            cols_water.append(col)
    
    return rows_water, cols_water 


def calibrate_area(rows_water, cols_water, calibration_x = 0, calibration_y = 0):
    calibrated_rows = np.array(rows_water) + calibration_y
    calibrated_cols = np.array(cols_water) + calibration_x

    return calibrated_rows, calibrated_cols

def get_RRS_H1_AC(cube_norm, calibrated_rows, calibrated_cols):
    RRS_H1_AC = []
    piksels = 0

    for row, col in zip(calibrated_rows, calibrated_cols):
        if not np.isnan(cube_norm[row, col]).any():
            RRS_H1_AC.append(cube_norm[row, col] / math.pi)
            piksels += 1

    print("Antall piksler i RRS_H1_AC:", piksels)     
    return RRS_H1_AC

def get_RRS_H1_and_H2_no_AC(l1d_cube, calibrated_rows, calibrated_cols):
    RRS_H1_no_AC = []
    piksels = 0

    for row, col in zip(calibrated_rows, calibrated_cols):
            RRS_H1_no_AC.append(l1d_cube[row, col,9:118])
            piksels += 1
    print("Antall piksler i RRS_H1_no_AC:", piksels)
    return RRS_H1_no_AC

def get_general_RRS_H2_AC(rho_wavelengths, AC_data_cube):
    rss_atmc = []
    for var in rho_wavelengths : 
        rss_atmc.append(AC_data_cube[var]/math.pi)

    return np.array(rss_atmc)     

def get_RRS_for_area_H2_AC(cal_rows, cal_columns, rss_data_general):
    rss_given = []
    for i in range(len(cal_rows)):
        piksel_values = rss_data_general[:, cal_rows[i], cal_columns[i]]
        has_nan = np.isnan(piksel_values)
        if has_nan.any() == False:
            rss_given.append(rss_data_general[:, cal_rows[i], cal_columns[i]])
        
    return np.array(rss_given)   

def automated_RRS_H1_AC(satobj_h1, l1d_cube, cube_norm, target_lat, target_lon, calibration_x, calibration_y):
    a_piksel_row, a_piksel_col = lat_long_to_piksel(target_lat, target_lon, satobj_h1.latitudes, satobj_h1.longitudes)
    row_1, col_1 = piksel_to_area(np.array(a_piksel_row), np.array(a_piksel_col), l1d_cube)
    water_rows, water_cols = area_within_watermask(row_1, col_1, satobj_h1.latitudes, satobj_h1.longitudes)
    print(water_cols)
    if water_cols != []:
        if all((x + calibration_x) < 684 for x in water_cols):
            cal_rows, cal_cols = calibrate_area(water_rows, water_cols, calibration_x, calibration_y)
        else: 
            RRS_H1_AC = get_RRS_H1_AC(cube_norm, water_rows, water_cols)
            print("this area not calibrated, here is target lat:", target_lat)
            return RRS_H1_AC
    else: 
        print(" this area is empty, here is target lat", target_lat)
        return water_cols
    RRS_H1_AC = get_RRS_H1_AC(cube_norm, cal_rows, cal_cols)
    piksles_to_area_check(np.array(a_piksel_row), np.array(a_piksel_col), l1d_cube, calibration_x, calibration_y)
    
    return RRS_H1_AC

def automated_RRS_NO_AC_H1(satobj_h1, l1d_cube, target_lat, target_lon, calibration_x = 0, calibration_y = 0):
    a_piksel_row, a_piksel_col = lat_long_to_piksel(target_lat, target_lon, satobj_h1.latitudes, satobj_h1.longitudes)
    row_1, col_1 = piksel_to_area(np.array(a_piksel_row), np.array(a_piksel_col), l1d_cube)
    water_rows, water_cols = area_within_watermask(row_1, col_1, satobj_h1.latitudes, satobj_h1.longitudes)

    if all((x + calibration_x) < 684 for x in water_cols):
        cal_rows, cal_cols = calibrate_area(water_rows, water_cols, calibration_x, calibration_y)
    else: 
        RRS_H1_no_AC = get_RRS_H1_and_H2_no_AC(l1d_cube, water_rows, water_cols)
        print("this area not calibrated, here is target lat:", target_lat)
        return RRS_H1_no_AC
    RRS_H1_no_AC = get_RRS_H1_and_H2_no_AC(l1d_cube, cal_rows, cal_cols)
    piksles_to_area_check(np.array(a_piksel_row), np.array(a_piksel_col), l1d_cube, calibration_x, calibration_y)
    
    return RRS_H1_no_AC

def automated_RRS_H2_AC(satobj_h2, l1d_cube, data_cube_corrected, rho_wavelengths, target_lat, target_lon, latitude, longitude): 
    ## sjekke opp i disse for skal ikke kalibreres, så sjekke om det funker å hente ut lat og long sånn her fra den korrektede kuben... 
    a_piksel_row, a_piksel_col = lat_long_to_piksel(target_lat, target_lon, latitude, longitude)
    row_1, col_1 = piksel_to_area(np.array(a_piksel_row), np.array(a_piksel_col), l1d_cube)
    water_rows, water_cols = area_within_watermask(row_1, col_1, latitude, longitude)
    #cal_rows, cal_cols = calibrate_area(water_rows, water_cols, calibration_x, calibration_y)

    RRS_H2_AC = get_RRS_for_area_H2_AC(water_rows, water_cols, get_general_RRS_H2_AC(rho_wavelengths, data_cube_corrected))
    #piksles_to_area_check(np.array(a_piksel_row), np.array(a_piksel_col), l1d_cube, calibration_x, calibration_y) ## evt fiks en så dette også kna plottes
    plt.imshow(data_cube_corrected['rho_w_460'])
    for i in range(len(a_piksel_row)):
        plt.scatter(a_piksel_col[i], a_piksel_row[i], color='red', s=5)

    return RRS_H2_AC

def automated_RRS_NO_AC_H2(satobj_h1, l1d_cube, target_lat, target_lon, latitude, longitude):
    a_piksel_row, a_piksel_col = lat_long_to_piksel(target_lat, target_lon, latitude, longitude)
    row_1, col_1 = piksel_to_area(np.array(a_piksel_row), np.array(a_piksel_col), l1d_cube)
    water_rows, water_cols = area_within_watermask(row_1, col_1, latitude, longitude)
    #cal_rows, cal_cols = calibrate_area(water_rows, water_cols, calibration_x, calibration_y)
    
    RRS_H1_no_AC = get_RRS_H1_and_H2_no_AC(l1d_cube, water_rows, water_cols)
    #piksles_to_area_check(np.array(a_piksel_row), np.array(a_piksel_col), l1d_cube, calibration_x, calibration_y)
    plt.imshow(l1d_cube[:,:, 50], origin='upper')
    for i in range(len(a_piksel_row)):
        plt.scatter(a_piksel_col[i], a_piksel_row[i], color='red', s=5)
    #plt.axis('off')
    #plt.savefig('../resultater/H2noAC/cloudsAt12_all_areasplotted.pdf', dpi=300, bbox_inches='tight')
    return RRS_H1_no_AC

#def get_RRS_H1_AC(cube_norm, calibrated_rows, calibrated_cols):
#    RRS_H1_AC = []
#    piksels = 0
#    np.array(calibrated_rows)
#    np.array(calibrated_cols)
#
#    ## check for is nan
#    for i in range(len(calibrated_rows)-1):
#        if(not np.isnan(cube_norm[i][i].any())):
#            RRS_H1_AC.append(cube_norm[calibrated_rows[i], calibrated_cols[i]] / math.pi)
#            piksels = piksels + 1
#
#    print("Antall piksler i RRS_H1_AC:", piksels)
#    return RRS_H1_AC


#def piksel_to_squareed_area(piksel_row, piksel_col): ## fjerne denne etter sammenligning med de andre funskjonene 
#    sorted_row = sorted(piksel_row)
#    sorted_col = sorted(piksel_col)
#
#    area_row = []
#    area_col = []
#
#    area_row.append(sorted_row[1])
#    area_row.append(sorted_row[-2])
#
#    area_col.append(sorted_col[1])
#    area_col.append(sorted_col[-2])
#
#    return area_row, area_col


## def get_RRS_H!_no_AC():
## def get_RRS_H2_AC(): ## dont need calibration
## def get_RRS_H2_no_AC(): ## dont need calibration

#def area_in_water(area_row, area_col, latitudes, longitudes): ## output not calibrated, but checked area is calibrated
#    # finn en måte og sjekk opp i dette på :))
#    rows_water = []
#    cols_water = []
#
#    for i in range(area_row[0], area_row[1] + 1):
#        for j in range(area_col[0], area_col[1] + 1):
#            if globe.is_ocean(latitudes[i][j], longitudes[i][j]):
#                rows_water.append(i)
#                cols_water.append(j)
#
#    return rows_water, cols_water

print("hei fra utformet bedre :)")

##checks!!!!
def piksles_to_area_check(piksel_row, piksel_col, datacube, calibration_x, calibration_y):
    image = datacube[:,:,100]
    #print(piksel_col[2], piksel_row[2])

    polygon_points = np.array([
        [piksel_col[0] + calibration_x, piksel_row[0] + calibration_y],  # Top-left
        [piksel_col[1] + calibration_x, piksel_row[1] + calibration_y],  # Top-right
        [piksel_col[2] + calibration_x, piksel_row[2] + calibration_y],  # Bottom-right
        [piksel_col[3] + calibration_x, piksel_row[3] + calibration_y]   # Bottom-left
    ], dtype=np.int32)

    image_np = image.values
    mask = np.zeros(image_np.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 1)

    # Extract pixel values
    pixels_in_area = image_np[mask == 1]
    
    image_with_poly = image.values.copy()
    image_bgr = cv2.cvtColor(image_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.polylines(image_bgr, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=2)
    
    img_min, img_max = image_np.min(), image_np.max()
    image_scaled = ((image_np - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    
    image_bgr = cv2.cvtColor(image_scaled, cv2.COLOR_GRAY2BGR)
    cv2.polylines(image_bgr, [polygon_points], isClosed=True, color=(255, 0, 0), thickness=10)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Originalbilde med polygon')
    plt.show()

    return 

def plot_RRS_piksles(datacube, water_rows, water_cols, calibration_x, calibration_y):
    plt.imshow(datacube[:,:, 100], origin='upper')
    for i in range(len(water_rows)):
        plt.scatter(water_cols[i]+calibration_x, water_rows[i]+ calibration_y, color='red', s=5)

    return plt

def automated_RRS_H1_AC_for_H2(satobj_h1, l1d_cube, cube_norm, target_lat, target_lon, latitudes, longitudes):
    a_piksel_row, a_piksel_col = lat_long_to_piksel(target_lat, target_lon, latitudes, longitudes)
    row_1, col_1 = piksel_to_area(np.array(a_piksel_row), np.array(a_piksel_col), l1d_cube)
    water_rows, water_cols = area_within_watermask(row_1, col_1, latitudes, longitudes)
    RRS_H1_AC = get_RRS_H1_AC(cube_norm, water_rows, water_cols)
    piksles_to_area_check(np.array(a_piksel_row), np.array(a_piksel_col), l1d_cube, 0, 0)
    
    return RRS_H1_AC