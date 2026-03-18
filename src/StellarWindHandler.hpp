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
 * but WIHTOUT ANY WARRANTY; without even the implied warranty of
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
#ifndef STELLARWINDHANDLER_HPP
#define STELLARWINDHANDLER_HPP

#include "DensitySubGridCreator.hpp"

/**
 * @brief Handler with useful functions.
 */
class StellarWindHandler {
private:

double _source_luminosities;

public:
  /**
   * @brief Constructor.
   *
   * @param grid DensityGrid to operate on.
   * @param opening_angle Opening angle that determines the accuracy of the
   * tree walk.
   */
  StellarWindHandler(){

        //constructor functionality

  }
   
   inline double get_sw_mass_loss_rate(double mass){

      static const std::vector<double> table_mass = { 5.897418435090001e+32, 3.94091048052e+32, 2.9619041341500004e+32, 2.37256574139e+32, 1.68294037743e+32, 1.18929695157e+32, 7.939154960100001e+31, 6.355975005900001e+31, 4.9686405444000005e+31, 3.9763978605000006e+31, 2.982752136e+31, 2.3865933429e+31, 1.7900813034e+31, 1.3922990055000003e+31 };
      static const std::vector<double> table_mass_loss_rate = { -4.125, -4.45, -4.691, -4.888, -5.22, -5.605, -6.149, -6.511, -6.977, -7.471, -7.976, -8.679, -9.973, -11.428 };
      double mass_loss_rate = 0.0;

      if (mass > table_mass.front()){
        mass_loss_rate = table_mass_loss_rate.front();
        mass_loss_rate = std::pow(10,mass_loss_rate);
        return mass_loss_rate;
      } else if (mass < table_mass.back()) {
        return 0.0;
      }

      // Find the interval containing x
      for (size_t i = 0; i < table_mass.size() - 1; ++i) {
          if (table_mass[i] >= mass && mass >= table_mass[i + 1]) {
              // Perform linear interpolation
              double x1 = table_mass[i];
              double x2 = table_mass[i + 1];
              double y1 = table_mass_loss_rate[i];
              double y2 = table_mass_loss_rate[i + 1];
              mass_loss_rate = y1 + (mass - x1) * (y2 - y1) / (x2 - x1);
              break;
          }
      }

      mass_loss_rate = std::pow(10.0, mass_loss_rate) * 1.989e30;
      return mass_loss_rate; //in kg
   };

   inline double get_sw_velocity(double mass){

      static const std::vector<double> table_mass = { 5.897418435090001e+32, 3.94091048052e+32, 2.9619041341500004e+32, 2.37256574139e+32, 1.68294037743e+32, 1.18929695157e+32, 7.939154960100001e+31, 6.355975005900001e+31, 4.9686405444000005e+31, 3.9763978605000006e+31, 2.982752136e+31, 2.3865933429e+31, 1.7900813034e+31, 1.3922990055000003e+31};
      static const std::vector<double> table_vels = { 3359.3377, 3368.0662, 3357.7034, 3338.8328, 3289.3316, 3211.7465, 3087.2741, 3004.4142, 2902.0406, 2801.616, 2664.5831, 2555.2617, 2413.9634, 1788.908};
      double vel = 0.0;

      if (mass > table_mass.front()){
        vel = table_vels.front();
        return vel;
      } else if (mass < table_mass.back()) {
        return 0.0;
      }

      // Find the interval containing x
      for (size_t i = 0; i < table_mass.size() - 1; ++i) {
          if (table_mass[i] >= mass && mass >= table_mass[i + 1]) {
              // Perform linear interpolation
              double x1 = table_mass[i];
              double x2 = table_mass[i + 1];
              double y1 = table_vels[i];
              double y2 = table_vels[i + 1];
              vel = y1 + (mass - x1) * (y2 - y1) / (x2 - x1);
              break;
          }
      }

      return vel; // in km/s
   };

   inline void inject_sw(
    DensitySubGridCreator<HydroDensitySubGrid>* grid_creator,
    Hydro& hydro,
    CoordinateVector<double> sw_loc,
    double mass,
    double timestep)
{
    double mdot  = get_sw_mass_loss_rate(mass);
    double vwind = get_sw_velocity(mass) * 1000.0;

    double dt_year = timestep / 31557600.0;

    double mass_to_inject   = mdot * dt_year;
    double energy_to_inject = 0.5 * mass_to_inject * vwind * vwind;

    // get local subgrid
    auto subgrid_it = grid_creator->get_subgrid(sw_loc);
    HydroDensitySubGrid &subgrid = *subgrid_it;

    // determine resolution scale
    double cell_vol = subgrid.get_cell(sw_loc).get_volume();
    double dx = std::pow(cell_vol, 1./3.);
    double r_inj = 4.0 * dx;

    // find cells inside sphere
    auto cells = grid_creator->cells_within_radius(sw_loc, r_inj);
    int numcells = cells.size();

    if (numcells == 0)
        return;

    double mass_per_cell   = mass_to_inject / numcells;
    double energy_per_cell = energy_to_inject / numcells;

    double density_add = mass_per_cell / cell_vol;

    for (const auto &c : cells) {

        auto sg_it = grid_creator->get_subgrid(c.first);
        HydroDensitySubGrid &sg = *sg_it;

        auto cellit = sg.hydro_begin() + c.second;

        auto &hydro_vars = cellit.get_hydro_variables();

        double rho_old = hydro_vars.get_primitives_density();
        double m_old   = hydro_vars.get_conserved_mass();

        hydro_vars.set_primitives_density(
            rho_old + density_add);

        hydro_vars.set_conserved_mass(
            m_old + mass_per_cell);

        hydro_vars.set_energy_term(
            hydro_vars.get_energy_term() + energy_per_cell);
    }

    return;
  }

  /**
   * @brief Destructor.
   */
  ~StellarWindHandler() {}
};

#endif // STELLARWINDHANDLER_HPP