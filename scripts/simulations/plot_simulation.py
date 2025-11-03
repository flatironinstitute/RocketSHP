import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

output_dir, run_id = sys.argv[1], sys.argv[2]
output_dir = Path(output_dir)

logger.info("Plotting results...")
data = np.loadtxt(output_dir / f"{run_id}_log.txt", delimiter=",")
step = data[:, 0]
potential_energy = data[:, 1]
temperature = data[:, 2]
volume = data[:, 3]

# Potential Energy
plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.plot(step, potential_energy, color="b", linewidth=1.5)
plt.xlabel("Step")
plt.ylabel("Potential Energy (kJ/mol)")

# Temperature
plt.subplot(3, 1, 2)
plt.plot(step, temperature, color="r", linewidth=1.5)
plt.xlabel("Step")
plt.ylabel("Temperature (K)")

# Volume
plt.subplot(3, 1, 3)
plt.plot(step, volume, color="g", linewidth=1.5)
plt.xlabel("Step")
plt.ylabel("Volume (nm³)")
plt.tight_layout()
plt.savefig(output_dir / f"{run_id}_md_plot.png", dpi=300)
plt.close()
