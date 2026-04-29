import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import ast
import yt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm

# -----------------------------
# Helper
# -----------------------------
def get_tb_grid(grid, subgriddim, gridsize):
    result = np.zeros(gridsize)
    cx, cy, cz = gridsize // subgriddim

    start = 0
    size = cx * cy * cz
    ix = iy = iz = 0

    while start + size <= grid.size:
        chunk = grid[start:start+size].reshape(cx, cy, cz)
        result[ix:ix+cx, iy:iy+cy, iz:iz+cz] = chunk

        start += size
        iz += cz
        if iz == gridsize[2]:
            iz = 0
            iy += cy
            if iy == gridsize[1]:
                iy = 0
                ix += cx

    return result

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Palatino Linotype",
    "mathtext.it": "Palatino Linotype:italic",
    "mathtext.bf": "Palatino Linotype:bold",

    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
})

width = 40  # pc (half-width)

# -----------------------------
# Settings
# -----------------------------
times = [2, 6, 10, 14]
folder = "/sharedscratch/jb450/CMacIonize/CMacIonize/SW_Test/"
save_path = "Png/2D"
os.makedirs(save_path, exist_ok=True)

pc = 3.085677581491367e16

field_names = ["Temperature", "Number Density", "Total Energy", "Velocity z"]
cmaps = ["hot", "inferno", "bone", "twilight_shifted"]

# -----------------------------
# Figure + layout
# -----------------------------
fig = plt.figure(figsize=(16, 16), constrained_layout=True)
gs = GridSpec(4, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.08])

axes = np.empty((4, 4), dtype=object)
cbar_axes = []

for row in range(4):
    for col in range(4):
        axes[row, col] = fig.add_subplot(gs[row, col])
    
    cbar_axes.append(fig.add_subplot(gs[row, 4]))

row_images = [None]*4

# -----------------------------
# Main loop
# -----------------------------
for col, time in enumerate(times):

    filename = f"disc_patch_{time:03d}.hdf5"

    with h5py.File(folder + filename, "r") as file:

        # --- metadata ---
        def read_attr(name):
            attr = file["/Parameters"].attrs[name]
            if isinstance(attr, bytes):
                attr = attr.decode("utf-8")
            return np.array(ast.literal_eval(attr))

        grid = read_attr("DensityGrid:number of cells")
        subgrids = read_attr("DensitySubGridCreator:number of subgrids")

        box = np.array(file["/Header"].attrs["BoxSize"])
        box_pc = box / pc
        box_x, box_y, box_z = box_pc / 2.0
        pix_size = box[0] / grid[0]

        # --- data ---
        fp = file["PartType0"]

        vz = get_tb_grid(np.array(fp["Velocities"])[:, 2], subgrids, grid)
        temp = get_tb_grid(fp["Temperature"], subgrids, grid)
        ntot = get_tb_grid(fp["NumberDensity"], subgrids, grid)
        nf   = get_tb_grid(fp["NeutralFractionH"], subgrids, grid)
        Etot = get_tb_grid(fp["TotalEnergy"], subgrids, grid)

        # units
        vz /= 1e3
        ntot /= 1e6
        vz[:, :, :grid[2]//2] *= -1

        # derived
        ionized_h = ntot * (1 - nf)
        alpha = 1.17e-13 * temp**(-0.942 - 0.030*np.log(temp))
        halpha = (alpha * ionized_h**2) / (4*np.pi) * pix_size**3

        bbox = np.array([
            [-box_x, box_x],
            [-box_y, box_y],
            [-box_z, box_z]
        ])

        ds = yt.load_uniform_grid(
            {
                ('gas','number_density'): (ntot, "cm**-3"),
                ('gas','temperature'): (temp, "K"),
                ('gas','velocity_z'): (vz, "km/s"),
                ('gas','total_energy'): (Etot, "J"),
            },
            grid,
            length_unit="pc",
            bbox=bbox
        )

        fields = [
            ('gas','temperature'),
            ('gas','number_density'),
            ('gas','total_energy'),
            ('gas','velocity_z')
        ]

        data_list = []

        for f in fields:
            proj = ds.proj(f, axis=1, weight_field=('gas','number_density'))

            frb = proj.to_frb(
                width=(2*box_x, 'pc'),
                resolution=(128, 128)
            )

            data_list.append(np.array(frb[f]))

    # -----------------------------
    # Plotting
    # -----------------------------
    for row in range(4):
        ax = axes[row, col]

        if row == 3:
            vmax = np.max(np.abs(data_list[row]))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        else:
            norm = None

        im = ax.imshow(data_list[row].T, extent=[-width, width, -width, width], cmap=cmaps[row], origin="lower", norm=norm)

        if col == 0:
            row_images[row] = im
            ax.set_ylabel(field_names[row])

        if row == 0:
            ax.set_title(f"{time/10} Myr")

        ax.set_box_aspect(1)
        ax.set_xlabel("x (pc)")
        ax.set_ylabel("z (pc)")

# -----------------------------
# Colorbars
# -----------------------------
for row in range(4):
    fig.colorbar(row_images[row], cax=cbar_axes[row])
    cbar_axes[row].set_ylabel(field_names[row])

# -----------------------------
# Save
# -----------------------------
fig.savefig(os.path.join(save_path, "4x4_panel.svg"))
plt.close(fig)