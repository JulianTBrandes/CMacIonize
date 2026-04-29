import matplotlib
matplotlib.use("Agg")

import h5py
import os, sys
import numpy as np
import ast

def get_tb_grid(grid,subgriddim,gridsize):
    result = np.zeros((gridsize[0],gridsize[1],gridsize[2]))

    cx = int(gridsize[0]/subgriddim[0])
    cy = int(gridsize[1]/subgriddim[1])
    cz = int(gridsize[2]/subgriddim[2])

    startchunk = 0
    endchunk = cx*cy*cz
    ix = 0
    iy = 0
    iz = 0
    while endchunk <= gridsize[0]*gridsize[1]*gridsize[2]:
        chunk = np.array(grid[startchunk:endchunk])
        result[ix:ix+cx,iy:iy+cy,iz:iz+cz] = chunk.reshape(cx,cy,cz)
        startchunk += cx*cy*cz
        endchunk += cx*cy*cz
        iz += cz
        if iz == gridsize[2]:
            iz = 0
            iy += cy
            if iy == gridsize[1]:
                iy = 0
                ix += cx

    return result

folders = ["/sharedscratch/jb450/CMacIonize/CMacIonize/SW_Test_3_No_SW/"]
save_path = "Slices"

times = np.arange(175, 226, 1)

# constants
pc = 3.085677581491367e16  # m

for i, folder in enumerate(folders):
    for t_ind, time in enumerate(times):

        # filename formatting
        if time < 10:
            addon = '00'
        elif time < 100:
            addon = '0'
        else:
            addon = ''

        filename = f'disc_patch_reference_{addon}{time}.hdf5'

        with h5py.File(folder + filename, "r") as file:

            # --- grid + geometry ---
            box = np.array(file["/Header"].attrs["BoxSize"])
            grid = np.array(ast.literal_eval(
                file["/Parameters"].attrs["DensityGrid:number of cells"].decode("utf-8")
            ))
            subgrids = np.array(ast.literal_eval(
                file["/Parameters"].attrs["DensitySubGridCreator:number of subgrids"].decode("utf-8")
            ))

            pix_size = box[0] / grid[0]  # cell size (m)

            # slice indices
            iy0 = grid[1] // 2
            iz0 = grid[2] // 2

            filepart = file["PartType0"]

            # --- raw fields ---
            velocities = np.array(filepart["Velocities"])
            velz = velocities[:, 2]

            vz = get_tb_grid(velz, subgrids, grid)
            temp = get_tb_grid(filepart["Temperature"], subgrids, grid)
            ntot = get_tb_grid(filepart["NumberDensity"], subgrids, grid)
            nf = get_tb_grid(filepart["NeutralFractionH"], subgrids, grid)

            # --- unit conversions ---
            vz /= 1e3  # km/s
            vz[:, :, 0:int(grid[2]/2)] *= -1  # NOTE: enforces symmetry

            ntot /= 1e6  # cm^-3

            # --- derived quantities ---
            neutral_h = ntot * nf
            ionized_h = ntot * (1 - nf)

            # --- column densities ---
            dx = pix_size * 100.0  # convert m → cm

            # along y → x–z plane
            col_ntot_y = np.sum(ntot, axis=1) * dx
            col_neutral_y = np.sum(neutral_h, axis=1) * dx
            col_ionized_y = np.sum(ionized_h, axis=1) * dx

            # along z → x–y plane
            col_ntot_z = np.sum(ntot, axis=2) * dx
            col_neutral_z = np.sum(neutral_h, axis=2) * dx
            col_ionized_z = np.sum(ionized_h, axis=2) * dx

            # --- slices ---

            # y = 0 (x–z plane)
            slice_y0 = {
                "number_density": ntot[:, iy0, :],
                "temperature": temp[:, iy0, :],
                "velocity_z": vz[:, iy0, :],
                "neutral_h": neutral_h[:, iy0, :],
                "ionized_h": ionized_h[:, iy0, :]
            }

            # z = 0 (x–y plane)
            slice_z0 = {
                "number_density": ntot[:, :, iz0],
                "temperature": temp[:, :, iz0],
                "velocity_z": vz[:, :, iz0],
                "neutral_h": neutral_h[:, :, iz0],
                "ionized_h": ionized_h[:, :, iz0]
            }

            # --- coordinates ---
            coords = filepart["Coordinates"]
            x = coords[:, 0].reshape(grid)
            y = coords[:, 1].reshape(grid)
            z = coords[:, 2].reshape(grid)

            x_axis = x[:, 0, 0] / pc
            y_axis = y[0, :, 0] / pc
            z_axis = z[0, 0, :] / pc

            # --- save output ---
            outdir = os.path.join(save_path)
            os.makedirs(outdir, exist_ok=True)

            outfile = os.path.join(outdir, f"slices_{addon}{time}.hdf5")

            with h5py.File(outfile, "w") as f:

                # slices
                grp_y = f.create_group("y0")
                for key, val in slice_y0.items():
                    grp_y.create_dataset(key, data=val)

                grp_z = f.create_group("z0")
                for key, val in slice_z0.items():
                    grp_z.create_dataset(key, data=val)

                # column densities
                col = f.create_group("column_density")

                col.create_dataset("ntot_y", data=col_ntot_y)
                col.create_dataset("neutral_y", data=col_neutral_y)
                col.create_dataset("ionized_y", data=col_ionized_y)

                col.create_dataset("ntot_z", data=col_ntot_z)
                col.create_dataset("neutral_z", data=col_neutral_z)
                col.create_dataset("ionized_z", data=col_ionized_z)

                # coordinates (now in pc)
                f.create_dataset("x", data=x_axis)
                f.create_dataset("y", data=y_axis)
                f.create_dataset("z", data=z_axis)

        print(f"Saved timestep {time} → {outfile}")