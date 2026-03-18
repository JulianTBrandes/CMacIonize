/*******************************************************************************
 * This file is part of CMacIonize
 * Copyright (C) 2016 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
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
 * @file PhotonSourceDistribution.hpp
 *
 * @brief Distribution functor for photon sources.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef PHOTONSOURCEDISTRIBUTION_HPP
#define PHOTONSOURCEDISTRIBUTION_HPP

#include "CoordinateVector.hpp"
#include "DensityGrid.hpp"
#include "HydroDensitySubGrid.hpp"
#include "DensitySubGridCreator.hpp"

/*! @brief Size of a variable that stores the number of photon sources. */
typedef uint_fast32_t photonsourcenumber_t;
class RandomGenerator;

/**
 * @brief General interface for photon source distribution functors.
 */
class PhotonSourceDistribution {
public:
  /**
   * @brief Virtual destructor.
   */
  virtual ~PhotonSourceDistribution() {}

  /**
   * @brief Get the number of sources contained within this distribution.
   *
   * The PhotonSourceDistribution will return exactly this number of valid
   * and unique positions by successive application of operator().
   *
   * @return Number of sources.
   */
  virtual photonsourcenumber_t get_number_of_sources() const = 0;

  /**
   * @brief Get a valid position from the distribution.
   *
   * @param index Index of the photon source, must be in between 0 and
   * get_number_of_sources().
   * @return CoordinateVector of a valid and photon source position (in m).
   */
  virtual CoordinateVector<> get_position(photonsourcenumber_t index) = 0;

  /**
   * @brief Get the weight of a photon source.
   *
   * @param index Index of the photon source, must be in between 0 and
   * get_number_of_sources().
   * @return Weight of the photon source, used to determine how many photons are
   * emitted from this particular source.
   */
  virtual double get_weight(photonsourcenumber_t index) const = 0;


  virtual double get_photon_weighting(photonsourcenumber_t index) const {return 0.0;}

  /**
   * @brief Get the total luminosity of all sources together.
   *
   * @return Total luminosity (in s^-1).
   */
  virtual double get_total_luminosity() const = 0;

  /**
   * @brief Update the distribution after the system moved to the given time.
   *
   * @param simulation_time Current simulation time (in s).
   * @return True if the distribution changed, false otherwise.
   */
  virtual bool update(const double simulation_time) { return false; }


  virtual bool update(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator, Hydro &hydro) {return false;}

  virtual bool update(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator) {return false;}

  virtual bool update(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator, double actual_timestep) {return false;}

  virtual bool update(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator, Hydro &hydro, double actual_timestep) {return false;}

  virtual void set_initial_velocity(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator, double timestep) {}
  virtual void float_sources(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator, double timestep) {}

  virtual void accrete_gas(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator, Hydro &hydro) {}

  virtual std::vector<CoordinateVector<double>> get_sink_positions() {
    std::vector<CoordinateVector<double>> vect;
    return vect;
  }

  virtual std::vector<double> get_sink_masses() {
    std::vector<double> vect;
    return vect;
  }

  /**
   * @brief Add stellar feedback to the given grid at the given time.
   *
   * Not all source distributions support stellar feedback.
   *
   * @param grid DensityGrid to operate on.
   * @param current_time Current simulation time (in s).
   */
  virtual void add_stellar_feedback(DensityGrid &grid,
                                    const double current_time) {}

  virtual void get_sne_radii(HydroDensitySubGrid &subgrid) {}

  virtual void get_sne_radii(DensitySubGridCreator< HydroDensitySubGrid > &grid_creator) {}

  /**
   * @brief Will the distribution do stellar feedback at the given time?
   *
   * Not all source distributions support stellar feedback.
   *
   * @param current_time Current simulation time (in s).
   * @return False, unless the implementation decides otherwise.
   */
  virtual bool do_stellar_feedback(const double current_time) const {
    return false;
  }

  /**
   * @brief Add stellar feedback to the given subgrid.
   *
   * Should only be called if add_stellar_feedback is called first.
   *
   * Not all source distributions support stellar feedback.
   *
   * @param subgrid DensitySubGrid to operate on.
   */
  virtual void add_stellar_feedback(HydroDensitySubGrid &subgrid) {}


  virtual void add_stellar_feedback(HydroDensitySubGrid &subgrid, Hydro &hydro) {}

  /**
   * @brief Finalise adding stellar feedback to a distributed grid.
   */
  virtual void done_stellar_feedback() {}

  virtual void add_stellar_wind_feedback(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator,Hydro &hydro) {}

  /**
   * @brief Write the distribution to the given restart file.
   *
   * @param restart_writer RestartWriter to use.
   */
  virtual void write_restart_file(RestartWriter &restart_writer) const {
    cmac_error(
        "Restarting is not supported for this PhotonSourceDistribution!");
  }

  virtual double get_photon_frequency(RandomGenerator &random_generator, photonsourcenumber_t index) {
    return 0.0;
  }
};

#endif // PHOTONSOURCEDISTRIBUTION_HPP
