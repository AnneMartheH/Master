import numpy as np
import matplotlib.pyplot as plt

# Depth array (0 to 15 meters)
depth = np.linspace(0, 15, 300)

# -------------------------
# Chlorophyll-a (Chl-a)
# -------------------------
max_chla = 7.51307297  # Max value between 3–4 m
target_chla_8 = 1      # Value at 8 m for the exponential start
chla = np.zeros_like(depth)

# Linear increase 0–3 m
mask1 = (depth <= 3)
chla[mask1] = (max_chla / 3) * depth[mask1]

# Constant max 3–4 m
mask2 = (depth > 3) & (depth <= 4)
chla[mask2] = max_chla

# Linear decrease 4–8 m (down to 1 at 8 m)
mask3 = (depth > 4) & (depth <= 8)
slope = (target_chla_8 - max_chla) / (8 - 4)
chla[mask3] = max_chla + slope * (depth[mask3] - 4)

# Exponential decrease 8–15 m
mask4 = (depth > 8)
chla[mask4] = target_chla_8 * np.exp(-0.3 * (depth[mask4] - 8))

# -------------------------
# CDOM
# -------------------------
max_cdom = 1.75  # Increased base value
cdom = np.zeros_like(depth)

# Constant max 0–4 m
mask_c1 = (depth <= 4)
cdom[mask_c1] = max_cdom

# Exponential decrease 4–15 m
mask_c2 = (depth > 4)
cdom[mask_c2] = max_cdom * np.exp(-0.25 * (depth[mask_c2] - 4))

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(6, 8))
plt.plot(chla, depth, label='Chl-a', color='green', linewidth=2)
plt.plot(cdom, depth, label='CDOM', color='brown', linewidth=2)

plt.gca().invert_yaxis()
plt.xlabel('Concentration chl-a [$\mu$g/L],  CDOM [$m^{-1}$]')  
plt.ylabel('Depth [m]')
plt.title('Illustration of D2 Water Profile')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('ecolight-s/water_profile_D2.pdf', dpi=300, bbox_inches='tight') 
#plt.show()

