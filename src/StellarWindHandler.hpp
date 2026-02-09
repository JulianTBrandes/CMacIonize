/*******************************************************************************
 * This file is part of CMacIonize
 * Copyright (C) 2018 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
 *
 * CMacIonize is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CMacIonize is distributed in the hope that it will be useful,
 * but WITOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with CMacIonize. If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

/**
 * @file StellarWindHandler.hpp
 *
 * @brief Stellar Wind handling functions.
 *
 * @author Julian Brandes (jb450@st-andrews.ac.uk)
 */
#ifndef StellarWindHandler_HPP
#define StellarWindHandler_HPP

#include <cmath>
#include "HydroIntegrator.hpp"

/**
 * @brief Handler with useful functions.
 */
class StellarWindHandler {
private:

    double _sw_energy;
    const double HydroIntegrator& hydrointegrator;

public:
  /**
   * @brief Constructor.
   *
   * @param grid DensityGrid to operate on.
   * @param opening_angle Opening angle that determines the accuracy of the
   * tree walk.
   */


  StellarWindHandler(const HydroIntegrator& hydrointegrator)
      : _hydrointegrator(hydrointegrator) {}

        inline double get_jeans_length(const DensityGrid::iterator& cell) const
    {
        const double rho =
            cell.get_hydro_variables().get_primitives_density();

        const double cs =
            _hydrointegrator.get_soundspeed(cell);

        return std::sqrt(M_PI * cs * cs / (G * rho));
    }

  }

  inline std::tuple<double,double,double,double> get_r_inj(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator,
                                         CoordinateVector<> sne_loc) {



      HydroDensitySubGrid &subgrid = *grid_creator->get_subgrid(sne_loc);

      double cell_vol =  subgrid.get_cell(sne_loc).get_volume();


      double dx = std::pow(cell_vol,1./3.);

      double r_run = 4*dx;
      double num_cells;
      std::vector<std::pair<uint_fast32_t,uint_fast32_t>> vec;


      vec = grid_creator->cells_within_radius(sne_loc,r_run);


      double mtot = 0.0;
      for (auto & pair : vec) {
        HydroDensitySubGrid &subgrid = *grid_creator->get_subgrid(std::get<0>(pair));
        mtot = mtot + (subgrid.hydro_begin() + std::get<1>(pair)).get_hydro_variables().get_conserved_mass();

      }
      if (mtot > 1.988e+33) {
        double inj_vol = 1.3333*3.14159265*std::pow(r_run,3.0);
        double rho = mtot/inj_vol;
        double nbar = 1.e-6*rho/1.67262192e-27;
        double r_st = 3.086e+16 * 19.1 * std::pow(_sw_energy*1.e-44,5./17.) * std::pow(nbar,-7./17);
        return std::make_tuple(r_run,r_st,nbar,268.);
      }

      while (mtot < 1.988e+33) {
        r_run = r_run+(0.25*dx);
        vec = grid_creator->cells_within_radius(sne_loc,r_run);
        mtot = 0.0;
        for (auto & pair : vec) {
          HydroDensitySubGrid &subgrid = *grid_creator->get_subgrid(std::get<0>(pair));
          double cell_mass = (subgrid.hydro_begin() + std::get<1>(pair)).get_hydro_variables().get_conserved_mass();
          mtot = mtot + cell_mass;
        }
      }
    
    num_cells = vec.size();
    double inj_vol = 1.3333*3.14159265*std::pow(r_run,3.0);
    double rho = mtot/inj_vol;
    double nbar = 1.e-6*rho/1.67262192e-27;
    double r_st = 3.086e+16 * 19.1 * std::pow(_sw_energy*1.e-44,5./17.) * std::pow(nbar,-7./17);
    return std::make_tuple(r_run, r_st, nbar, num_cells);

   }



   inline void inject_sne(HydroDensitySubGrid &subgrid, Hydro &hydro, CoordinateVector<double> position,
         double r_inj, double r_st, double nbar, int numcells) {

      for (auto cellit = subgrid.hydro_begin();
           cellit != subgrid.hydro_end(); ++cellit) {

           CoordinateVector<> cellpos = cellit.get_cell_midpoint();

          // is cell within injeciton radius of SNe?
           if ((cellpos - position).norm() < r_inj) {
                if (cellit.get_hydro_variables().get_primitives_density() == 0) {
                    //dont add energy to cell without mass...
                    return;
                }
             double dx = std::pow(cellit.get_volume(),1./3.);
             if (r_st < 4.*dx) {

//Not resolving ST radius, do momentum injection
              CoordinateVector<> vel_prior =
                       cellit.get_hydro_variables().get_primitives_velocity();

                // Blondin et al
               double mom_to_inj = 2.6e5*std::pow(nbar,-2./17) * std::pow(_sw_energy*1.e-44,16./17.);
               // Msol km/s to kg m/s
               mom_to_inj = mom_to_inj * 2.e30 * 1.e3;

               double m_tot = (nbar*1e6*1.67e-27)*(4.*3.14159265*std::pow(r_inj,3)/3.);

               double vel_to_inj = mom_to_inj/m_tot;

               CoordinateVector<> direction = (cellpos-position)/((cellpos-position).norm());

               CoordinateVector<> vel_new = vel_prior + vel_to_inj*direction;

               cellit.get_hydro_variables().set_primitives_velocity(vel_new);


           //    double density = cellit.get_hydro_variables().get_primitives_density();

             //  double xH = cellit.get_ionization_variables().get_ionic_fraction(ION_H_n);


              // double pressure = 8254.397014*1.e4*density*2./(1.+xH);

              //cellit.get_ionization_variables().set_temperature(1.e4);

             // cellit.get_hydro_variables().set_primitives_pressure(pressure);

              hydro.set_conserved_variables(cellit.get_hydro_variables(), cellit.get_volume());

             }
             else {
               cellit.get_hydro_variables().set_energy_term(_sw_energy/numcells);
             }
           }

        }




   }

  /**
   * @brief Destructor.
   */
  ~StellarWindHandler() {}
};

#endif // StellarWindHandler_HPP