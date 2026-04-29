import matplotlib
matplotlib.use("Agg")

import h5py
import os,sys
import numpy as np
import ast
import matplotlib.pyplot as plt
import yt
import re
import matplotlib as mpl
import matplotlib.lines as mlines
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

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

plt.rcParams.update({
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

folder = "/sharedscratch/jb450/CMacIonize/CMacIonize/SW_Test/"
save_path = "Png_2"

os.makedirs(os.path.join(save_path, "2D"), exist_ok=True)

times = np.arange(0, 6, 1) # fiducial sims to 500 Myr


# constants
pc = 3.085677581491367*10**16  #m
kpc=3.085677581491367*10**16 * 10**3 #m

Msol = 2*10**30 # kg
yr = 365.25*24*60*60 # seconds

for t_ind, time in enumerate(times):

    if time < 10:
        addon = '00'
    elif time<100:
        addon = '0'
    else:
        addon = ''
    filename = 'disc_patch_'+addon+str(time)+'.hdf5'

    with h5py.File(folder+filename) as file:
        #get simulation box dimension in SI and pc
        box = np.array(file["/Header"].attrs["BoxSize"])
        box_pc = box/pc
        grid = ast.literal_eval(file["/Parameters"].attrs["DensityGrid:number of cells"].decode("utf-8") )
        grid = np.array(grid)
        print('grid = ', grid)
        subgrids = ast.literal_eval(file["/Parameters"].attrs["DensitySubGridCreator:number of subgrids"].decode("utf-8"))
        subgrids = np.array(subgrids)
        pix_size = box[0]/grid[0]
        # use real box size from file
        box_x = box_pc[0] / 2
        box_y = box_pc[1] / 2
        box_z = box_pc[2] / 2
        cell_x = 128
        cell_y = cell_x
        cell_z = 768
        domain = np.array([[-cell_x, cell_x], [-cell_y, cell_y], [-cell_z, cell_z]])
        domain = np.array([[-box_x, box_x], [-box_y, box_y], [-box_z, box_z]])
        filepart = file['PartType0']
        print(list(filepart.keys()))

        velocities = np.array(filepart['Velocities'])
        velz = velocities[:,2]
        vz = get_tb_grid(velz,subgrids,grid) # print these mgb
        coords = filepart['Coordinates']
        x = coords[:,0].reshape(grid)
        y = coords[:,1].reshape(grid)
        z = coords[:,2].reshape(grid)
        
        xs = x[:,63,:]
        zs = z[:,63,:]

        # --- Reconstruct fields ---
        temp = get_tb_grid(filepart['Temperature'], subgrids, grid)
        ntot = get_tb_grid(filepart['NumberDensity'], subgrids, grid)
        nf   = get_tb_grid(filepart['NeutralFractionH'], subgrids, grid)
        vz   = get_tb_grid(velz, subgrids, grid)
        # Etot = get_tb_grid(filepart['TotalEnergy'], subgrids, grid)

        # --- Unit conversions ---
        vz /= 1e3              # m/s → km/s
        ntot /= 1e6            # m^-3 → cm^-3

        # --- Derived quantities (correct cell volume) ---
        dx = box[0] / grid[0]
        dy = box[1] / grid[1]
        dz = box[2] / grid[2]
        cell_volume = dx * dy * dz

        alpha = 1.17e-13 * temp**(-0.942 - 0.030*np.log(temp))
        halpha = (alpha * (((1 - nf) * ntot)**2)) / (4*np.pi) * cell_volume

        neutral_h = ntot * nf
        ionized_h = ntot * (1 - nf)

        # --- yt dataset ---
        bbox = np.array([
            [-box_x, box_x],
            [-box_y, box_y],
            [-box_z, box_z]
        ])

        data_fields = {
            ('gas', 'number_density'): (ntot, "cm**-3"),
            ('gas', 'temperature'): (temp, "K"),
            ('gas', 'neutral_fraction'): (nf, ""),
            ('gas', 'velocity_z'): (vz, "km/s"),
            ('gas', 'halpha'): (halpha, "cm**3/s"),
            ('gas', 'neutral_hydrogen'): (neutral_h, "m**-3"),
            ('gas', 'ionized_hydrogen'): (ionized_h, "m**-3"),
            #('gas', 'total_energy'): (Etot, "J")
        }

        ds = yt.load_uniform_grid(
            data_fields,
            grid,
            length_unit="pc",
            bbox=bbox,
            nprocs=1
        )

        nx = grid[0]
        ny = grid[1]
        nz = grid[2]

        center = ds.domain_center

        slc = ds.slice('z', center[2])

        frb = slc.to_frb(
            width=ds.domain_width[0],
            height=ds.domain_width[1],
            resolution=(nx, ny),
            center=center
        )

        # --- Extract images ---
        temp_img_data = np.array(frb['gas', 'temperature'])
        ntot_img_data = np.array(frb['gas', 'number_density'])
        # Etot_img_data = np.array(frb['gas', 'total_energy'])
        vel_z_data    = np.array(frb['gas', 'velocity_z'])
        ionizedh_data = np.array(frb['gas', 'ionized_hydrogen'])
        neutralh_data = np.array(frb['gas', 'neutral_hydrogen'])

        #   ds = [ds_ntot, ds_temp, ds_vz, ds_neutralh, ds_ionizedh, ds_halpha, ds_nf] #, ds_column_H, ds_column_HI, ds_column_HII]
        cmaps = ['inferno', 'hot', 'viridis', 'bone', 'pink', 'gist_heat', 'twilight_shifted_r']
        
        plot_extent = [
            -box_pc[0]/2, box_pc[0]/2,
            -box_pc[1]/2, box_pc[1]/2
        ]

        fig = plt.figure(figsize=(20, 20))

        gs = GridSpec(3, 4, figure = fig)#,  width_ratios=[2, 2, 2, 1, 1, 1, 1, 0.1] )
        mo1 = fig.add_subplot(gs[:,0])
        mo2 = fig.add_subplot(gs[:,1])
        mo3 = fig.add_subplot(gs[:,2])
        mo4 = fig.add_subplot(gs[:,3])

        mo1.set_box_aspect(1)
        mo2.set_box_aspect(1)
        mo3.set_box_aspect(1)
        mo4.set_box_aspect(1)

        mo1.set_xlabel('x (pc)')
        mo2.set_xlabel('x (pc)')
        mo3.set_xlabel('x (pc)')
        mo4.set_xlabel('x (pc)')
        mo1.set_ylabel('y (pc)')

        mo1_s = mo1.imshow(temp_img_data.T, cmap='hot', extent=plot_extent, origin='lower', aspect='auto')#,norm=mcolors.LogNorm(vmin=1e2, vmax=1e6))
        cb1 = fig.colorbar(mo1_s, ax=mo1, label='Temp (K)', orientation='horizontal', location='top', shrink=0.83, pad=0.001)
        mo2_s = mo2.imshow(neutralh_data.T, cmap='inferno', extent=plot_extent, origin='lower', aspect='auto', norm=mcolors.LogNorm(vmin=1e-10, vmax=1e0))
        cb2 = fig.colorbar(mo2_s, ax=mo2, label=r'$n_{H^{0}}$ ($cm^{-3}$)', orientation='horizontal', location='top', shrink=0.83, pad=0.001)
        mo3_s = mo3.imshow(ionizedh_data.T, cmap='bone', extent=plot_extent, origin='lower', aspect='auto', norm=mcolors.LogNorm(vmin=1e-7, vmax=1e-3))
        cb3 = fig.colorbar(mo3_s, ax=mo3, label=r'$n_{H^{+}}$ ($cm^{-3}$)', orientation='horizontal', location='top', shrink=0.83, pad=0.001)
        mo4_s = mo4.imshow(vel_z_data.T, cmap='twilight_shifted', extent=plot_extent, origin='lower', aspect='auto', vmin=-1.1*np.max(np.abs(vel_z_data)), vmax=1.1*np.max(np.abs(vel_z_data)))
        cb4 = fig.colorbar(mo4_s, ax=mo4, label=r'$v_{z}$ ($km \ s^{-1}$)', orientation='horizontal', location='top', shrink=0.83, pad=0.001)


        cb1.set_label(r'Temperature (K)', labelpad=20)
        cb2.set_label(r'$n_{H^{0}}$ ($cm^{-3}$)', labelpad=20)
        cb3.set_label(r'$n_{H^{+}}$ ($cm^{-3}$)', labelpad=20)
        cb4.set_label(r'$v_{z}$ ($km \ s^{-1}$)', labelpad=20)


        fig.suptitle(str(time)+' Myr')
        filepath = os.path.join(save_path,'Slice','Slice_Plot_'+addon+str(time)+'.png')
        fig.savefig(filepath)
        plt.close(fig)
        print("Box (pc):", box_pc)
            


'''

#if you want slices:


                slc = ds.slice('y', coord=0.0)
                p = yt.SlicePlot(ds, 'y', ('gas', 'temperature'))
                p.set_cmap(('gas', 'temperature'), 'hot')
                p.save('../mo_shs_mass/slice_temp.png')

                x_coords = np.linspace(-box_x, box_x, 128 )
                z_coords = np.linspace(-box_z, box_z, 768 )
                X, Z = np.meshgrid(x_coords, z_coords)

                frb = slc.to_frb(width=(6, 'kpc'), height=(1, 'kpc'), resolution=(768,128), center=(0,0,0))
                temp_img_data = np.array(frb['gas', 'temperature'])
                ntot_img_data = np.array(frb['gas', 'number_density'])
                halpha_data = np.array(frb['gas', 'halpha'])
                ionizedh_data = np.array(frb['gas', 'ionized_hydrogen'])
                neutralh_data = np.array(frb['gas', 'neutral_hydrogen'])


             #   ds = [ds_ntot, ds_temp, ds_vz, ds_neutralh, ds_ionizedh, ds_halpha, ds_nf] #, ds_column_H, ds_column_HI, ds_column_HII]
                cmaps = ['inferno', 'hot', 'viridis', 'bone', 'pink', 'gist_heat', 'twilight_shifted_r'] # gist_earth, magma (like inferno) YlGnbu, Spectral (blue larger)


                print(f"Time Step: {time} Myr")
                print(f"Temp data shape: {temp_img_data.shape}, Min/Max: {np.min(temp_img_data):.2e}, {np.max(temp_img_data):.2e}")
                plot_extent = [-box_x, box_x, -box_z, box_z]

                mo1_s = mo1.imshow(temp_img_data.T, cmap='hot', extent=plot_extent, origin='lower', aspect='auto',norm=mcolors.LogNorm(vmin=100, vmax=1e7))
                cb1 = fig.colorbar(mo1_s, ax=mo1, label='Temp (K)', orientation='horizontal', location='top', shrink=0.83, pad=0.001)
                mo2_s = mo2.imshow(neutralh_data.T, cmap='inferno', extent=plot_extent, origin='lower', aspect='auto', norm=mcolors.LogNorm(vmin=1e-10, vmax=1e2))
                cb2 = fig.colorbar(mo2_s, ax=mo2, label=r'$n_{H^{0}}$ ($cm^{-3}$)', orientation='horizontal', location='top', shrink=0.83, pad=0.001)
                mo3_s = mo3.imshow(ionizedh_data.T, cmap='bone', extent=plot_extent, origin='lower', aspect='auto', norm=mcolors.LogNorm(vmin=1e-8, vmax=1e0))
                cb3 = fig.colorbar(mo3_s, ax=mo3, label=r'$n_{H^{+}}$ ($cm^{-3}$)', orientation='horizontal', location='top', shrink=0.83, pad=0.001)

                cb1.set_label('Temp (K)', labelpad=20)
                cb2.set_label(r'$n_{H^{0}}$ ($cm^{-3}$)', labelpad=20)
                cb3.set_label(r'$n_{H^{+}}$ ($cm^{-3}$)', labelpad=20)

'''