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


folders = ["/sharedscratch/jb450/CMacIonize/CMacIonize/SW_Test/"]
save_path = "Png"

os.makedirs(os.path.join(save_path, "2D"), exist_ok=True)
os.makedirs(os.path.join(save_path, "Number_Density"), exist_ok=True)
os.makedirs(os.path.join(save_path, "Temperature"), exist_ok=True)

times = np.arange(0, 31, 1) # fiducial sims to 500 Myr


# constants
pc = 3.085677581491367*10**16  #m
kpc=3.085677581491367*10**16 * 10**3 #m

Msol = 2*10**30 # kg
yr = 365.25*24*60*60 # seconds

for i, folder in enumerate(folders):
    for t_ind, time in enumerate(times):
        fig = plt.figure(layout='constrained', figsize=(60,30))

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
        mo1.set_ylabel('z (pc)')

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
            box_x = 40
            box_y = box_x
            box_z = box_x
            cell_x = 64
            cell_y = cell_x
            cell_z = 64
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

            temp = get_tb_grid(filepart['Temperature'],subgrids,grid)
            ntot = get_tb_grid(filepart['NumberDensity'],subgrids,grid)
            nf = get_tb_grid(filepart['NeutralFractionH'], subgrids, grid)
            vz /= (10**3) #convert to m/s
            vz[:,:,0:int(grid[2]/2)] = -(vz[:,:,0:int(grid[2]/2)])
            ntot /= (10**6) #convert to cm^-3

            mass = get_tb_grid(filepart['Mass'],subgrids,grid)
            Etot = get_tb_grid(filepart['TotalEnergy'],subgrids,grid)

            alpha = 1.17 *10**-13 * temp**(-0.942-0.030*np.log(temp))# data["stream", "Temperature"]**(-0.942-0.030*np.log(data["stream", "Temperature"]))  cm^3/second
            halpha = (alpha * (((1-nf)*ntot)**2))/(4*np.pi) *pix_size**3 # (((1-data["stream", "Neutral Fraction"])*data["Number Density"])**2))/(4*np.pi) cm^3 s^-1 m^-3 m^3 
            neutral_h = ntot * nf
            ionized_h = ntot * (1 - nf)
            
            temp_sl = temp[:,0,:] #
            print(np.shape(temp_sl))
            ntot_sl = ntot[:,63,:] #np.sum(ntot, axis=1)
            halpha_proj = np.sum(halpha, axis=1)
            
            bbox = np.array([[-40, 40], [-40, 40], [-40, 40]])
            ntot_yt = {('gas', 'number_density'): (ntot, "cm**-3")}
            nf_yt = {('gas', 'neutral_fraction'): (nf, "")}
            temp_yt = {('gas', 'temperature'): (temp, "K")}
            vz_yt = {('gas', 'velocity_z'): (vz, "km/s")} 
            halpha_yt = {('gas', 'halpha'): (halpha, "cm**3/s")} # this unit is not correct ...
            Neutral_H_yt = {('gas', 'neutral_hydrogen'): (neutral_h, "m**-3")}
            Ionized_H_yt = {('gas', 'ionized_hydrogen'): (ionized_h, "m**-3")}
            mass_yt = {('gas', 'mass'): (mass, "kg")}
            Etot_yt = {('gas', 'total_energy'): (Etot, "J")}
            data_fields = {**ntot_yt, **temp_yt, **nf_yt, **vz_yt, **halpha_yt, **Neutral_H_yt, **Ionized_H_yt, **mass_yt, **Etot_yt}

            # this will make projection plots weighted by number density:
            ds = yt.load_uniform_grid(data_fields, grid, length_unit =("pc"), bbox=bbox, nprocs=1)
            slc = ds.proj(('gas', 'number_density'), axis=1, weight_field=('gas', 'number_density'))
            slc2 = ds.proj(('gas', 'temperature'), axis=1, weight_field=('gas', 'number_density'))
            slc3 = ds.proj(('gas', 'total_energy'), axis=1, weight_field=('gas', 'number_density'))
            slc4 = ds.proj(('gas', 'velocity_z'), axis=1, weight_field=('gas', 'number_density'))
            # p = yt.ProjectionPlot(ds, 'y', ('gas', 'temperature')) # make single projection plot 
            # p.save('../mo_shs_mass/proj.png')

            frb = slc.to_frb(width=(40, 'pc'), height=(40, 'pc'), resolution=(128,128), center=(0,0,0))
            frb2 = slc2.to_frb(width=(40, 'pc'), height=(40, 'pc'), resolution=(128,128), center=(0,0,0))
            frb3 = slc3.to_frb(width=(40, 'pc'), height=(40, 'pc'), resolution=(128,128), center=(0,0,0))
            frb4 = slc4.to_frb(width=(40, 'pc'), height=(40, 'pc'), resolution=(128,128), center=(0,0,0))
            
            mass_img_data = np.array(frb['gas', 'mass'])
            temp_img_data = np.array(frb2['gas', 'temperature'])
            Etot_img_data = np.array(frb3['gas', 'total_energy'])
            vel_z_data = np.array(frb4['gas', 'velocity_z'])
            #halpha_data = np.array(frb['gas', 'halpha'])
            ntot_img_data = np.array(frb['gas', 'number_density'])
            

            #   ds = [ds_ntot, ds_temp, ds_vz, ds_neutralh, ds_ionizedh, ds_halpha, ds_nf] #, ds_column_H, ds_column_HI, ds_column_HII]
            cmaps = ['inferno', 'hot', 'viridis', 'bone', 'pink', 'gist_heat', 'twilight_shifted_r'] # gist_earth, magma (like inferno) YlGnbu, Spectral (blue larger)
            # Blue-Red
            
            print(f"Time Step: {time} Myr")
            print(f"Temp data shape: {temp_img_data.shape}, Min/Max: {np.min(temp_img_data):.2e}, {np.max(temp_img_data):.2e}")
            plot_extent = [-box_x, box_x, -box_z, box_z]

            mo1_s = mo1.imshow(temp_img_data.T, cmap='hot', extent=plot_extent, origin='lower', aspect='auto')#,norm=mcolors.LogNorm(vmin=1e2, vmax=1e6))
            cb1 = fig.colorbar(mo1_s, ax=mo1, label='Temp (K)', orientation='horizontal', location='top', shrink=0.83, pad=0.001)
            mo2_s = mo2.imshow(ntot_img_data.T, cmap='inferno', extent=plot_extent, origin='lower', aspect='auto')#,vmin=-1.1*np.max(np.abs(ntot_img_data)), vmax=1.1*np.max(np.abs(ntot_img_data)))#, norm=mcolors.LogNorm(vmin=1, vmax=10))
            cb2 = fig.colorbar(mo2_s, ax=mo2, label=r'Number Density (cm^{-3})', orientation='horizontal', location='top', shrink=0.83, pad=0.001)
            mo3_s = mo3.imshow(Etot_img_data.T, cmap='bone', extent=plot_extent, origin='lower', aspect='auto')#,vmin=-1.1*np.max(np.abs(Etot_img_data)), vmax=1.1*np.max(np.abs(Etot_img_data)))#, norm=mcolors.LogNorm(vmin=1e34, vmax=1e41))
            cb3 = fig.colorbar(mo3_s, ax=mo3, label=r'Total Energy (J)', orientation='horizontal', location='top', shrink=0.83, pad=0.001)
            mo4_s = mo4.imshow(vel_z_data.T, cmap='twilight_shifted', extent=plot_extent, origin='lower', aspect='auto', vmin=-1.1*np.max(np.abs(vel_z_data)), vmax=1.1*np.max(np.abs(vel_z_data)))
            cb4 = fig.colorbar(mo4_s, ax=mo4, label=r'$v_{z}$ ($km \ s^{-1}$)', orientation='horizontal', location='top', shrink=0.83, pad=0.001)


            cb1.set_label(r'Temperature (K)', labelpad=20)
            cb2.set_label(r'Number Density (cm^{-3})', labelpad=20)
            cb3.set_label(r'Total Energy (J)', labelpad=20)
            cb4.set_label(r'$v_{z}$ ($km \ s^{-1}$)', labelpad=20)


            fig.suptitle(str(time)+' Myr')
            filepath = os.path.join(save_path,'2D','2D_Plot_'+addon+str(time)+'.png')
            fig.savefig(filepath)
            plt.close(fig)
            

            center = [0, 0, 0]

            # Define spherical region
            sp = ds.sphere(center, (40, "pc"))
            r = sp["index", "radius"].to("pc")

            r_min = r.min()
            r_max = r.max()

            nbins = 128
            equal_volume = False  # <-- toggle

            u = np.linspace(0, 1, nbins + 1)

            if equal_volume:
                # equal-volume spacing (uniform in r^3)
                r_edges = (r_min**3 + (r_max**3 - r_min**3) * u) ** (1/3)
            else:
                # linear spacing in r
                r_edges = r_min + (r_max - r_min) * u

            r_edges = ds.arr(r_edges, "pc")

            profile = yt.create_profile(
                sp,
                ("index","radius"),
                fields=[("gas","temperature"), ("gas","number_density"),("gas","mass")],
                weight_field=("gas","number_density"),
                override_bins={("index","radius"): r_edges}
            )

            r = profile.x.to("pc")
            temp = profile[("gas", "temperature")]
            ntot = profile[("gas", "number_density")]
            mass = profile[("gas", "mass")]

            # Temperature
            fig1, ax1 = plt.subplots()

            ax1.plot(r, temp)
            ax1.set_title(f"{time} Myr")
            ax1.set_xlabel("Radius (pc)")
            ax1.set_ylabel("Temperature (K)")
            ax1.set_yscale("log")
            ax1.set_xscale("linear")
            ax1.set_ylim(10,10**9)
            filepath = os.path.join(save_path,"Temperature",'Radial_Plot_Temp_'+addon+str(time)+'.png')
            fig1.savefig(filepath, dpi=300)
            plt.close(fig1)


            # Number Density
            fig2, ax2 = plt.subplots()

            ax2.plot(r, ntot)
            ax2.set_title(f"{time} Myr")
            ax2.set_xlabel("Radius (pc)")
            ax2.set_ylabel("Number Density (cm$^{-3}$)")
            ax2.set_yscale("log")
            ax2.set_xscale("linear")
            ax2.set_ylim(10**-6,30)
            filepath = os.path.join(save_path,"Number_Density","Radial_Plot_Number_Density_"+addon+str(time)+'.png')
            fig2.savefig(filepath, dpi=300)
            plt.close(fig2)

            # Mass
            fig3, ax3 = plt.subplots()

            ax3.plot(r, mass)
            ax3.set_title(f"{time} Myr")
            ax3.set_xlabel("Radius (pc)")
            ax3.set_ylabel("Mass (kg)")
            ax3.set_yscale("log")
            ax3.set_xscale("linear")
            #ax3.set_ylim(10**-6,30)
            filepath = os.path.join(save_path,"Mass","Radial_Plot_Mass_"+addon+str(time)+'.png')
            fig3.savefig(filepath, dpi=300)
            plt.close(fig3)

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