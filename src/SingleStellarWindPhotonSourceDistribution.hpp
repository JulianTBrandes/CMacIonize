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
 * @file SingleStellarWindPhotonSourceDistribution.hpp
 *
 * @brief PhotonSourceDistribution without UV sources that mimicks stellar wind.
 *
 * Used to test the hydrodynamical impact of a single supernova explosion on
 * an ISM with some equation of state.
 *
 * @author Julian Brandes (jb450@st-andrews.ac.uk)
 */
#ifndef SINGLESTELLARWINDPHOTONSOURCEDISTRIBUTION_HPP
#define SINGLESTELLARWINDPHOTONSOURCEDISTRIBUTION_HPP

#include "Log.hpp"
#include "ParameterFile.hpp"
#include "PhotonSourceDistribution.hpp"
#include "RestartReader.hpp"
#include "RestartWriter.hpp"
#include "StellarWindHandler.hpp"
#include "SupernovaHandler.hpp"

/**
 * @brief PhotonSourceDistribution without UV sources that mimicks a single
 * supernova explosion with an adjustable energy output.
 */
class SingleStellarWindPhotonSourceDistribution
    : public PhotonSourceDistribution {
private:
  /*! @brief Position of the supernova explosion (in m). */
  const CoordinateVector<> _position;

  /*! @brief Life time of the source before it explodes (in s). */
  const double _lifetime;

  /*! @brief Luminosity of the source before it goes supernova (in s^-1). */
  const double _luminosity;

  /*! @brief Energy of the supernova exposion (in J). */
  const double _energy;

  /*! @brief Mass of the star (in Msol). */
  double _mass;

  /*! @brief Flag signalling if the supernova already exploded. */
  bool _has_exploded;
  bool _inform_dist;

  double _r_inj;
  double _r_st;

  double _nbar;

  double _num_cells_injected;

  double _simulation_time = 0;

  StellarWindHandler *swhandler;
  SupernovaHandler *novahandler;

public:
  /**
   * @brief Constructor.
   *
   * @param position Position of the supernova explosion (in m).
   * @param lifetime Life time of the source before it explodes (in s).
   * @param luminosity Ionising luminosity of the source before it goes
   * supernova (in s^-1).
   * @param energy Energy of the supernova explosion (in J).
   * @param log Log to write logging info to.
   */
  inline SingleStellarWindPhotonSourceDistribution(
      const CoordinateVector<> position, const double lifetime, const double luminosity,
      const double energy,const double mass, Log *log = nullptr)
      : _position(position), _lifetime(lifetime), _luminosity(luminosity), _energy(energy), 
      _mass(mass), _has_exploded(false),_inform_dist(false) {


      swhandler = new StellarWindHandler();
      novahandler = new SupernovaHandler(_energy);


    if (log != nullptr) {
      log->write_status(
          "Set up SingleStellarWindPhotonSourceDistribution at position [",
          _position.x(), " m, ", _position.y(), " m, ", _position.z(),
          " m], with life time ", _lifetime, " s, ionising luminosity ",
          _luminosity, " s^-1, with SN energy ", _energy, " J, and with Mass ",_mass," Msol.");
    }
  }

  /**
   * @brief ParameterFile constructor.
   *
   * Parameters are:
   *  - position: Position of the supernova explosion (default: [0. m, 0. m, 0.
   *    m])
   *  - lifetime: Life time of the source before it explodes (default: 10. Myr)
   *  - luminosity: Ionising luminosity of the source before it goes supernova
   *    (default: 1.e49 s^-1)
   *  - energy: Energy of the supernova explosion (default: 1.e51 erg)
   *
   * @param params ParameterFile to read from.
   * @param log Log to write logging info to.
   */
  SingleStellarWindPhotonSourceDistribution(ParameterFile &params,
                                          Log *log = nullptr)
    : SingleStellarWindPhotonSourceDistribution(
          params.get_physical_vector<QUANTITY_LENGTH>(
             "PhotonSourceDistribution:position", "[0. m, 0. m, 0. m]"),
          params.get_physical_value<QUANTITY_TIME>(
              "PhotonSourceDistribution:lifetime", "4.0 Myr"),
          params.get_physical_value<QUANTITY_FREQUENCY>(
              "PhotonSourceDistribution:luminosity", "0.0 s^-1"),
          params.get_physical_value<QUANTITY_ENERGY>(
              "PhotonSourceDistribution:energy", "1.e51 erg"),
          params.get_physical_value<QUANTITY_MASS>(
              "PhotonSourceDistribution:mass", "30 Msol"),
          log) {}

  /**
   * @brief Virtual destructor.
   */
  virtual ~SingleStellarWindPhotonSourceDistribution() {
    delete swhandler;
    delete novahandler;
  }

  /**
   * @brief Get the number of sources contained within this distribution.
   *
   * The PhotonSourceDistribution will return exactly this number of valid
   * and unique positions by successive application of operator().
   *
   * @return Number of sources.
   */
  virtual photonsourcenumber_t get_number_of_sources() const {
    if (_luminosity > 0. && !_has_exploded) {
      return 1;
    } else {
      return 0;
    }
  }

  /**
   * @brief Get a valid position from the distribution.
   *
   * @param index Index of the photon source, must be in between 0 and
   * get_number_of_sources().
   * @return CoordinateVector of a valid and photon source position (in m).
   */
  virtual CoordinateVector<> get_position(photonsourcenumber_t index) {
    return _position;
  }

  /**
   * @brief Get the weight of a photon source.
   *
   * @param index Index of the photon source, must be in between 0 and
   * get_number_of_sources().
   * @return Weight of the photon source, used to determine how many photons are
   * emitted from this particular source.
   */
  virtual double get_weight(photonsourcenumber_t index) const { return 1.; }

  /**
   * @brief Get the total luminosity of all sources together.
   *
   * @return Total luminosity (in s^-1).
   */
  virtual double get_total_luminosity() const {
    if (!_has_exploded) {
      return _luminosity;
    } else {
      return 0.0;
    }
  }


  /**
   * @brief Will the distribution do stellar feedback at the given time?
   *
   * @param current_time Current simulation time (in s).
   * @return True if the star has not exploded yet and its lifetime has been
   * exceeded.
   */
  virtual bool do_stellar_feedback(const double current_time) const {
    return (!_has_exploded && current_time >= _lifetime);
  }

  virtual void get_sne_radii(DensitySubGridCreator< HydroDensitySubGrid > &grid_creator) {



        double r_inj,r_st,nbar,num_inj;


        std::tie(r_inj,r_st,nbar,num_inj) = novahandler->get_r_inj(&grid_creator,_position);

         _r_inj=r_inj;
         _r_st = r_st;
         _nbar = nbar;
         _num_cells_injected = num_inj;


       
  }

  virtual void add_stellar_feedback(HydroDensitySubGrid &subgrid, Hydro &hydro) {



      novahandler->inject_sne(subgrid, hydro, _position, _r_inj,_r_st,_nbar,_num_cells_injected);

    
  }

  /**
   * @brief Finalise adding stellar feedback to a distributed grid.
   */
  virtual void done_stellar_feedback() { 

    std::cout << "\n SNe INJECTION HERE: R_inj = " << _r_inj << " R_st = " <<  _r_st
       << " num_cells = " <<  _num_cells_injected << " nbar = "  << _nbar << "\n";
    
    _has_exploded = true; }

  /**
   * @brief Update the distribution after the system moved to the given time.
   *
   * @param simulation_time Current simulation time (in s).
   * @return True if the distribution changed, false otherwise.
   */
  virtual bool update(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator, Hydro &hydro, double actual_timestep) override {
    _simulation_time += actual_timestep;
    if (!_has_exploded && _simulation_time <= _lifetime) {
      swhandler->inject_sw(
            grid_creator,
            hydro,
            _position,
            _mass,
            actual_timestep);
      return false;
    }
    if (_has_exploded && !_inform_dist) {
      _inform_dist = true;
      return false;
    } else {
      // make sure the PhotonSource is warned if the number of sources is
      // zero
      return false;
    }
  }

  /**
   * @brief Write the distribution to the given restart file.
   *
   * @param restart_writer RestartWriter to use.
   */
  virtual void write_restart_file(RestartWriter &restart_writer) const {

    _position.write_restart_file(restart_writer);
    restart_writer.write(_lifetime);
    restart_writer.write(_luminosity);
    restart_writer.write(_energy);
    restart_writer.write(_mass);
    restart_writer.write(_has_exploded);
    restart_writer.write(_simulation_time);
  }

  /**
   * @brief Restart constructor.
   *
   * @param restart_reader Restart file to read from.
   */
  inline SingleStellarWindPhotonSourceDistribution(RestartReader &restart_reader)
      : _position(restart_reader), _lifetime(restart_reader.read< double >()),
        _luminosity(restart_reader.read< double >()),
        _energy(restart_reader.read< double >()),
        _mass(restart_reader.read< double >()),
        _has_exploded(restart_reader.read< bool >()) {
          _simulation_time = restart_reader.read<double>();
          swhandler = new StellarWindHandler();
          novahandler = new SupernovaHandler(_energy);
        }
};

#endif // SingleStellarWindPhotonSourceDistribution_HPP
