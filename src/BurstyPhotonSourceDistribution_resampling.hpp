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
 * @file BurstyPhotonSourceDistribution.hpp
 *
 * @brief Bursty PhotonSourceDistribution.
 *
 * @author Meg Blackburn (mgb27@st-andrews.ac.uk)
 */
#ifndef BURSTYPHOTONSOURCEDISTRIBUTION_HPP
#define BURSTYPHOTONSOURCEDISTRIBUTION_HPP

#include "Log.hpp"
#include "ParameterFile.hpp"
#include "PhotonSourceDistribution.hpp"
#include "RandomGenerator.hpp"
#include "DensitySubGridCreator.hpp"
#include "SupernovaHandler.hpp"
#include "WMBasicPhotonSourceSpectrum.hpp"
#include "PowerLawPhotonSourceSpectrum.hpp"
#include "Pegase3PhotonSourceSpectrum.hpp"
#include "PhotonSourceSpectrum.hpp"

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <unistd.h>
#include <vector>



/**
 * @brief Disc patch PhotonSourceDistribution.
 */
class BurstyPhotonSourceDistribution : public PhotonSourceDistribution {
private:
  /*! @brief Lifetime of a source (in s). */
  const double _star_formation_rate;
   // edit mgb 19.09.2025
// const double bursty_star_formation_rate; 
  /*! @brief Bursty SFR flag*/
  const bool _bursty_sfr_flag;
  /*! @brief Length of Burst*/
  const double _length_of_burst;
  /*! @brief Time of burst Peak (in Myr). */
  const double _time_of_burst_peak;
  /*! @brief Burst Amplitude */
  const double _burst_amplitude; // how much more star formation during burst
  // end of mgb edit 

  const double _peakiness_fraction; 
  /*! @brief peakiness fraction */
  const double _scale_height_peak;
  /*! scale height for peak driving */
  const double _type1_sne_scale_height;
  /*! scale height for type 1 supernovae*/
  const bool _type1_flag;
  /*! flag for including type I supernovae */
  const double _kennicutt_schmidt_index;
  /*! index within the Kennicutt-Schmidt relation */
  const bool _restart_flag;
  /*! restart flag */
  const double _restart_time_myr;
  /*! restart time given in Myr converted to seconds */
  const double _M_init; // mgb edit 27.01.2026
  /*! initial mass for restart */
  const double _M_init_unscaled_for_sfr; // mgb edit 27.01.2026
  /*! initial mass unscaled for sfr, for restart */

  /*! @brief Update time interval (in s). */
  const double _update_interval;

  /*! @brief Positions of the sources (in m). */
  std::vector< CoordinateVector<> > _source_positions;

  /*! @brief Remaining lifetime of the sources (in s). */
  std::vector< double > _source_lifetimes;
  std::vector< double > _source_masses;
  std::vector< double > _source_luminosities;

  std::vector<int> _to_delete;


// std::vector< CoordinateVector<> > _source_positions_moving; // mgb edit from SinkSource
  std::vector< CoordinateVector<> > _source_velocities; // mgb edit 16.10.2025

  std::vector < double > _cum_imf;
  std::vector < double > _mass_range;

  /*! @brief Output file for the sources (if applicable). */
  std::ofstream *_output_file;

  std::ofstream *_output_file2;

  /*! @brief Number of updates since the start of the simulation. */
  uint_fast32_t _number_of_updates;

  /*! @brief Indices of the sources (if output is enabled). */
  std::vector< uint_fast32_t > _source_indices;

  std::vector<int> _spectrum_index;
  std::vector<PhotonSourceSpectrum*> _all_spectra;

  std::vector<double> stellarMasses = {57.95, 46.94, 38.08, 34.39, 30.98, 28.0, 25.29, 22.90, 20.76, 18.80, 17.08, 15.55};
  std::vector<double> temperatures = {44852, 42857, 40862, 39865, 38867, 37870, 36872, 35874, 34877, 33879, 32882, 31884};

  std::vector<double> avail_temps = {32000, 34000, 34000, 35000, 36000, 37000, 39000, 39000, 40000, 41000,42000,43000,44000,45000};

  /*! @brief Index of the next source to add (if output is enabled). */
  uint_fast32_t _next_index;


  std::vector< CoordinateVector<> > _to_do_feedback;

  std::vector< double > _r_inj;
  std::vector< double > _r_st;
  std::vector< double > _num_cells_injected;
  std::vector< double > _nbar;

  const double _sne_energy = 1.e44;

  // defining constants: mgb edit 12.11.2025 
  const double unit_Myr = 3.15576e13;
  const double unit_yr = 3.154e+7;
  const double unit_kpc = 3.086e+19;
  const double unit_Msol = 1.988e30;

  const double _lum_adjust;

  double _excess_mass = 0;

  const double _scale_height;

  const double _peak_fraction;

  /*! @brief Pseudo-random number generator. */

  double init_running_mass;
  double init_running_mass_unscaled_for_sfr;
  uint_fast32_t _num_sne = 0;

  double _holmes_time;
  double _holmes_sh;
  double _holmes_lum;
  uint_fast32_t _number_of_holmes;


  bool _read_file;
  std::string _filename;
  std::string _source_filename;
  std::string _total_luminosity_filename;
  double _time;

  int type1done = 0;


  double _total_time = 0.;

  bool _holmes_added = false;

  double _last_sf = 0.;

  RandomGenerator _random_generator;

  SupernovaHandler *novahandler;

  Log *_log;


    static double kroupa_imf(double mass) {
      if (mass > 0.5) {
        return std::pow(mass,-2.3);
      } else if (mass > 0.08){
        return 2*std::pow(mass,-1.3);
      } else {
        return 25*std::pow(mass,-0.3);
      }
    }

    double integral(double (*f)(double), double a, double b, int n) {
    double step = (b - a) / n;  // width of each small rectangle
    double area = 0.0;  // signed area
    for (int i = 0; i < n; i ++) {
        area += f(a + (i + 0.5) * step) * step; // sum up each small rectangle
    }
    return area;
    }

    double get_single_mass(std::vector<double> mass_range,
           std::vector<double> cum_imf, double rand_num) {

       int Nup = mass_range.size()-1;
       int Nlow=0;
       int mid=(Nup + Nlow)/2;
       while(Nup - Nlow > 1){
         mid=(Nup + Nlow)/2;
         if (rand_num > cum_imf[mid]){
           Nlow = mid;
         } else {
           Nup = mid;
         }
       }

     return (mass_range[Nup] + mass_range[Nlow])/2.0;


    }

// Function to perform linear interpolation for descending xVals
double interpolate(double x, const std::vector<double>& xVals, const std::vector<double>& yVals) {
    // Ensure inputs are valid
    if (xVals.size() != yVals.size() || xVals.empty()) {
        throw std::invalid_argument("Invalid input: xVals and yVals must have the same size and cannot be empty.");
    }


    if (x > xVals.front()){
      return yVals.front();
    } else if (x < xVals.back()) {
      return yVals.back();
    }

    // Find the interval containing x
    for (size_t i = 0; i < xVals.size() - 1; ++i) {
        if (xVals[i] >= x && x >= xVals[i + 1]) {
            // Perform linear interpolation
            double x1 = xVals[i];
            double x2 = xVals[i + 1];
            double y1 = yVals[i];
            double y2 = yVals[i + 1];
            return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
        }
    }

    // If we reach here, x was not in any valid interval
    throw std::logic_error("Could not interpolate: x is not within any interval.");
}


size_t findClosestIndex(double value, const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("The values vector cannot be empty.");
    }

    size_t closestIndex = 0;
    double minDifference = std::abs(value - values[0]);

    for (size_t i = 1; i < values.size(); ++i) {
        double difference = std::abs(value - values[i]);
        if (difference < minDifference) {
            minDifference = difference;
            closestIndex = i;
        }
    }

    return closestIndex;
}

    double lum_from_mass(double mass) {

      std::vector<double> tablemasses= {57.95, 46.94, 38.08, 34.39, 30.98, 28.0, 25.29, 22.90, 20.76, 18.80, 17.08, 15.55};
      std::vector<double> tablelums = {49.64,49.44,49.22,49.10,48.99,48.88,48.75,48.61,48.44,48.27,48.06,47.88};
      double lum = 0.0;

      if (mass > tablemasses.front()){
        lum = tablelums.front();
        lum = std::pow(10,lum);
        lum = lum*_lum_adjust;
        return lum;
      } else if (mass < tablemasses.back()) {
        return 0.0;
      }


      // Find the interval containing x
      for (size_t i = 0; i < tablemasses.size() - 1; ++i) {
          if (tablemasses[i] >= mass && mass >= tablemasses[i + 1]) {
              // Perform linear interpolation
              double x1 = tablemasses[i];
              double x2 = tablemasses[i + 1];
              double y1 = tablelums[i];
              double y2 = tablelums[i + 1];
              lum = y1 + (mass - x1) * (y2 - y1) / (x2 - x1);
          }
      }

      lum = std::pow(10,lum);
      lum = lum*_lum_adjust;
      return lum;

    }



public:
  /**
   * @brief Constructor.
   *
   * @param source_lifetime Lifetime of a source (in s).
   * @param source_luminosity Ionising luminosity of a single source (in s^-1).
   * @param average_number Average number of sources at any given time.
   * @param anchor_x x component of the anchor of the rectangular disk (in m).
   * @param sides_x x side length of the rectangular disk (in m).
   * @param anchor_y  y component of the anchor of the rectangular disk (in m).
   * @param sides_y y side length of the rectangular disk (in m).
   * @param origin_z Origin of the Gaussian disk height distribution (in m).
   * @param scaleheight_z Scale height of the Gaussian disk height distribution
   * (in m).
   * @param seed Seed for the pseudo-random number generator.
   * @param update_interval Time interval in between successive source
   * distribution updates (in s).
   * @param starting_time Start time of the simulation. The distribution is
   * evolved forward in time to this point before it is used (in s).
   * @param output_sources Should the source positions be written to a file?
   */
  inline BurstyPhotonSourceDistribution(
      const double star_formation_rate,
      const bool bursty_sfr_flag,
      const double length_of_burst, // mgb edit 19.09.2025
      const double time_of_burst_peak, // mgb edit 19.09.2025
      const double burst_amplitude, // mgb edit 19.09.2025
      const double peakiness_fraction, // mgb edit 11.11.2025
      const double scale_height_peak, // mgb edit 11.11.2025
      const double type1_sne_scale_height, // mgb edit 11.11.2025
      const bool type1_flag, // mgb edit 12.11.2025
      const double kennicutt_schmidt_index, // mgb edit 12.11.2025 
      const bool restart_flag, // mgb edit 21.11.2025
      const double restart_time_myr, // mgb edit 08.12.2025
      const double M_init, // mgb edit 27.01.2026 
      const double M_init_unscaled_for_sfr, // mgb edit 27.01.2026
    //  const bool moving_sources_flag, // mgb 16.10.2025
      const int_fast32_t seed, const double update_interval,
      const double starting_time, bool output_sources = false,
      const double sne_energy = 1.e44,
      const double lum_adjust=1.0,
      const double scale_height=0.0,
      const double peak_fraction=0.5,
      const double holmes_time=0.0,
      const double holmes_sh=3e18,
      const double holmes_lum=5e46,
      const uint_fast32_t number_of_holmes=200,
      const bool read_file=false,
      const std::string filename="sources.txt",
      const std::string source_filename="Bursty_source_positions.txt",
      const std::string total_luminosity_filename="TotalLuminosity.txt",
      const double time=100.,
      Log *log=nullptr)
      : _star_formation_rate(star_formation_rate), _bursty_sfr_flag(bursty_sfr_flag), _length_of_burst(length_of_burst), _time_of_burst_peak(time_of_burst_peak),
        _burst_amplitude(burst_amplitude), _peakiness_fraction(peakiness_fraction),
        _scale_height_peak(scale_height_peak), _type1_sne_scale_height(type1_sne_scale_height), _type1_flag(type1_flag), 
        _kennicutt_schmidt_index(kennicutt_schmidt_index), _restart_flag(restart_flag), _restart_time_myr(restart_time_myr),
        _M_init(M_init), _M_init_unscaled_for_sfr(M_init_unscaled_for_sfr),
        _update_interval(update_interval),
        _output_file(nullptr), _number_of_updates(1), _next_index(0),
        _sne_energy(sne_energy), _lum_adjust(lum_adjust), _scale_height(scale_height),
        _peak_fraction(peak_fraction),_holmes_time(holmes_time),
        _holmes_sh(holmes_sh),_holmes_lum(holmes_lum),_number_of_holmes(number_of_holmes),
        _read_file(read_file), _filename(filename), _source_filename(source_filename), _total_luminosity_filename(total_luminosity_filename), _time(time),
        _random_generator(seed), _log(log){

    novahandler = new SupernovaHandler(_sne_energy);

    

    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(32000,25,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(34000,25,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(34000,25,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(35000,40,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(36000,25,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(37000,25,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(39000,25,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(39000,25,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(40000,25,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(41000,40,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(42000,40,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(43000,40,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(44000,40,log));
    _all_spectra.push_back(new WMBasicPhotonSourceSpectrum(45000,40,log));
    _all_spectra.push_back(new Pegase3PhotonSourceSpectrum(1e10,0.02,log));


    // form cumulative IMF
    double imf_start = 8.0;
    double imf_end = 120;

    double full_area = integral(kroupa_imf, imf_start, imf_end, 10000);

    uint_fast32_t range_length = 10000;
    for (uint_fast32_t i=0; i< range_length; ++i){
      double step = (imf_end-imf_start)/range_length;
      _mass_range.push_back(imf_start + step*i);
      double part_integral = integral(kroupa_imf, imf_start,imf_start + step*i,10000);
      _cum_imf.push_back(part_integral/full_area);
    }

    if (output_sources) {
      if (_restart_flag == true) { // don't think this if statement is needed anymore mgb 28.01.2026
        _output_file = new std::ofstream(_source_filename);
        *_output_file << "#time (s)\tx (m)\ty (m)\tz (m)\tevent\tindex\tluminosity\tMass\ttype\n";
        _output_file->flush();

        _output_file2 = new std::ofstream(_total_luminosity_filename);
        *_output_file2 << "simulation time (s)\trestart time (s)\tlum (s^-1)\tnumsne\tSFR_base (Msol Myr-1)\tSFR_KS (Msol Myr^-1 kpc^-2)\tSFR_KS (kg s^-1 kpc^-2)\tM_init\tM_init_unscaled_for_sfr\n";
        _output_file2->flush(); 
      } else {
        _output_file = new std::ofstream(_source_filename);
        *_output_file << "#time (s)\tx (m)\ty (m)\tz (m)\tevent\tindex\tluminosity\tMass\ttype\n";
        _output_file->flush();

        _output_file2 = new std::ofstream(_total_luminosity_filename);
        *_output_file2 << "simulation time (s)\trestart time (s)\tlum (s^-1)\tnumsne\tSFR_base (Msol Myr-1)\tSFR_KS (Msol Myr^-1 kpc^-2)\tSFR_KS (kg s^-1 kpc^-2)\tM_init\tM_init_unscaled_for_sfr\n";
        _output_file2->flush();
      }

    }

    if (_read_file){
      std::ifstream file;
      file.open(_filename);
      if (!file.is_open()) {
        cmac_error("Could not open file \"%s\"!", _filename.c_str());
      } else {
        std::cout << "Opened file - " << _filename << std::endl;
      }

      double time_val,posx,posy,posz,luminosity,mass;
      int event,index;

      std::string dummyLine,star_type;

      std::getline(file, dummyLine);

      time_val = 0.0;

      while (!file.eof() && time_val <= time) {
        file >> time_val >> posx >> posy >> posz >> event >> index >> luminosity >> mass >> star_type;

        if (event == 2) {
          _to_delete.push_back(index);
        }
      }

    file.close();

    file.open(_filename);


    std::getline(file, dummyLine);

   // CoordinateVector<> cell_vel = hydro_variables.get_primitives_velocity() // mgb 16.10.2025

    double a0z = 9.955209529401348; // mgb note need to find out what these values are 12.11.2025 
    double a1z = -3.3370109454102326;
    double a2z = 0.8116654874025604;
    time_val = 0.0;
    while (!file.eof() && time_val <= time) {
      file >> time_val >> posx >> posy >> posz >> event >> index >> luminosity >> mass >> star_type;
      if (event == 1) {
        if (std::find(_to_delete.begin(), _to_delete.end(), index) == _to_delete.end()) {
            
            _source_positions.push_back(CoordinateVector<double>(posx,posy,posz));

            _source_luminosities.push_back(luminosity);
            
            double lifetime = a0z + a1z*std::log10(mass) + a2z*(std::log10(mass)*std::log10(mass));
            lifetime = std::pow(10.0,lifetime);
            lifetime = lifetime*unit_yr; //3.154e+7;
            lifetime -= (_time - time_val);
            _source_lifetimes.push_back(lifetime);
          //  _source_velocities.push_back(gas_vel); // mgb 16.10.2025
          //  _source_positions_moving.push_back(midpoint) // mgb 16.10.2025 - taken out as don't think we want midpoints like sinkstar does
            _source_indices.push_back(_next_index);
            _source_masses.push_back(mass);
            if (star_type == "HOLMES") {
              _spectrum_index.push_back(14);
              if (_output_file != nullptr) {
                *_output_file << _total_time << "\t" << posx << "\t" << posy
                          << "\t" << posz << "\t1\t"
                          << _source_indices.back() << "\t"
                          << _source_luminosities.back() << "\t"
                          << mass << "\t"
                          << "HOLMES\n";
              }
            } else {
              double interpolatedTemp = interpolate(mass, stellarMasses, temperatures);
              size_t closestIndex = findClosestIndex(interpolatedTemp, avail_temps);
              std::cout << "Adding star of mass " << mass << " temp of " << interpolatedTemp << " for spec index " << closestIndex << std::endl;
              _spectrum_index.push_back(closestIndex);
              if (_output_file != nullptr) {
                *_output_file << _total_time << "\t" << posx << "\t" << posy
                          << "\t" << posz << "\t1\t"
                          << _source_indices.back() << "\t"
                          << _source_luminosities.back() << "\t"
                          << mass << "\t"
                          << "OSTAR\n";
              }
            }
            ++_next_index;

      }
    }
  }

  file.close();
  }

  }



  // -----------------------------------------------

  /**
   * @brief ParameterFile constructor.
   *
   * Parameters are:
   *  - source lifetime: Lifetime of a source (default: 20. Myr)
   *  - source luminosity: Ionising luminosity of a single source
   *    (default: 1.e48 s^-1)
   *  - average number of sources: Average number of sources (default: 24)
   *  - anchor x: X position of the anchor of the 2D disc (default: -1. kpc)
   *  - sides x: X side length of the 2D disc (default: 2. kpc)
   *  - anchor y: Y position of the anchor of the 2D disc (default: -1. kpc)
   *  - sides y: Y side length of the 2D disc (default: 2. kpc)
   *  - origin z: Origin of the exponential disc profile in the z direction
   *    (default: 0. pc)
   *  - scaleheight z: Vertical scale height of the exponential disc profile
   *    (default: 63. pc)
   *  - random seed: Random seed used to initialize the random generator that
   *    is used to sample the individual positions (default: 42)
   *  - update interval: Time interval in between successive distribution
   *    updates (default: 0.1 Myr)
   *  - starting time: Starting time of the simulation. The distribution is
   *    evolved forward in time to this point before it is used
   *    (default: 0. Myr)
   *  - output sources: Whether or not to write the source positions to a file
   *    (default: false)
   *
   * @param params ParameterFile to read from.
   * @param log Log to write logging info to.
   */
  BurstyPhotonSourceDistribution(ParameterFile &params, Log *log = nullptr)
      : BurstyPhotonSourceDistribution(
            params.get_physical_value< QUANTITY_MASS_RATE >(
                "PhotonSourceDistribution:star formation rate", "0.01 Msol yr^-1"),
            params.get_value< bool >("PhotonSourceDistribution:bursty sfr flag", 
                                       false),
            params.get_physical_value< QUANTITY_LENGTH_OF_BURST >( 
                "PhotonSourceDistribution:length of burst", "10.0 Myr"), // edit mgb 19.09.2025
            params.get_physical_value< QUANTITY_TIME_OF_BURST_PEAK >(
                "PhotonSourceDistribution:time of burst peak", "80.0 Myr"), // edit mgb 19.09.2025 - may need to change unit to yr
            params.get_value< double >("PhotonSourceDistribution:burst amplitude", 5.0), // edit mgb 19.09.2025
            params.get_value< double >("PhotonSourceDistribution:peakiness fraction", 1.0), // mgb edit 11.11.2025x
            params.get_physical_value< QUANTITY_LENGTH >( // mgb edit 11.11.2025
                "PhotonSourceDistribution:scale height peak", "200 pc"), 
            params.get_physical_value< QUANTITY_LENGTH >( // mgb edit 11.11.2025
                "PhotonSourceDistribution:type1 sne scale height", "325 pc"),
            params.get_value< bool >("PhotonSourceDistribution:type1 flag", true), // edit mgb 12.11.2025
            params.get_value< double >("PhotonSourceDistribution:kennicutt schmidt index", 1.4), // mgb edit 12.11.2025
            params.get_value< bool >("TaskBasedRadiationHydrodynamicsSimulation:restart flag", false), // mgb edit 21.11.2025
            params.get_physical_value< QUANTITY_TIME >("PhotonSourceDistribution:restart time myr", "0 Myr"), // mgb edit 08.12.2025
            params.get_physical_value< QUANTITY_MASS >("PhotonSourceDistribution:initial mass", "0.0 Msol"), // mgb edit 27.01.2026
            params.get_physical_value< QUANTITY_MASS >("PhotonSourceDistribution:initial mass unscaled for sfr", "0.0 Msol"), // mgb edit 27.01.2026
         //   params.get_value< bool >("PhotonSourceDistribution: moving sources flag", false),
            params.get_value< int_fast32_t >(
                "PhotonSourceDistribution:random seed", 42),
            params.get_physical_value< QUANTITY_TIME >(
                "PhotonSourceDistribution:update interval", "0.1 Myr"),
            params.get_physical_value< QUANTITY_TIME >(
                "PhotonSourceDistribution:starting time", "0. Myr"),
            params.get_value< bool >("PhotonSourceDistribution:output sources",
                                     false),
            params.get_physical_value< QUANTITY_ENERGY > (
                "PhotonSourceDistribution:supernova energy", "1.e51 erg"),
            params.get_value< double >("PhotonSourceDistribution:luminosity adjust",1.0),
            params.get_physical_value<QUANTITY_LENGTH> (
              "PhotonSourceDistribution:scale height","0.0 m"),
            params.get_value< double >("PhotonSourceDistribution:peak fraction",0.5),
            params.get_physical_value<QUANTITY_TIME> (
                "PhotonSourceDistribution:holmes time","50 Myr"),
            params.get_physical_value<QUANTITY_LENGTH>(
                "PhotonSourceDistribution:holmes height","700 pc"),
            params.get_physical_value<QUANTITY_FREQUENCY>(
                "PhotonSourceDistribution:holmes luminosity","5e46 s^-1"),
            params.get_value<double>("PhotonSourceDistribution:number of holmes",200),
            params.get_value<bool>("PhotonSourceDistribution:read file",false),
            params.get_value<std::string>("PhotonSourceDistribution:filename","SourceFile.txt"),
            params.get_value<std::string>("PhotonSourceDistribution:source filename","Bursty_source_positions.txt"),
            params.get_value<std::string>("PhotonSourceDistribution:total luminosity filename","TotalLuminosity.txt"),
            params.get_physical_value<QUANTITY_TIME>("PhotonSourceDistribution:time","0.0 Myr"),
            log) {}

  /**
   * @brief Virtual destructor.
   */
  virtual ~BurstyPhotonSourceDistribution() {}

  /**
   * @brief Get the number of sources contained within this distribution.
   *
   * The PhotonSourceDistribution will return exactly this number of valid
   * and unique positions by successive application of operator().
   *
   * @return Number of sources.
   */
  virtual photonsourcenumber_t get_number_of_sources() const {
    return _source_positions.size();
  }

  /**
   * @brief Will the distribution do stellar feedback at the given time?
   *
   * @param current_time Current simulation time (in s).
   * @return True if the star has not exploded yet and its lifetime has been
   * exceeded.
   */
  virtual bool do_stellar_feedback(const double current_time) const {
    return (_to_do_feedback.size() > 0);
  }

  virtual void get_sne_radii(DensitySubGridCreator< HydroDensitySubGrid > &grid_creator) {

       for (uint_fast32_t i = 0; i < _to_do_feedback.size(); ++i) {

        double r_inj,r_st,nbar,num_inj;


        std::tie(r_inj,r_st,nbar,num_inj) = novahandler->get_r_inj(&grid_creator,_to_do_feedback[i]);

         _r_inj.push_back(r_inj);
         _r_st.push_back(r_st);
         _nbar.push_back(nbar);
         _num_cells_injected.push_back(num_inj);


       }
  }

  virtual void add_stellar_feedback(HydroDensitySubGrid &subgrid, Hydro &hydro) {

    for (uint_fast32_t i = 0; i < _to_do_feedback.size(); ++i) {

      novahandler->inject_sne(subgrid, hydro, _to_do_feedback[i], _r_inj[i],_r_st[i],_nbar[i],_num_cells_injected[i]);

    }
  }

  virtual void done_stellar_feedback() {

    for (uint_fast32_t i=0; i<_to_do_feedback.size();i++) {

    std::cout << "\n SNe INJECTION HERE: R_inj = " << _r_inj[i] << " R_st = " <<  _r_st[i]
       << " num_cells = " <<  _num_cells_injected[i] << " nbar = "  << _nbar[i] << "\n";
    }

    _to_do_feedback.clear();
    _r_inj.clear();
    _r_st.clear();
    _num_cells_injected.clear();
    _nbar.clear();

  }



  /**
   * @brief Get a valid position from the distribution.
   *
   * @param index Index of the photon source, must be in between 0 and
   * get_number_of_sources().
   * @return CoordinateVector of a valid and photon source position (in m).
   */
  virtual CoordinateVector<> get_position(photonsourcenumber_t index) {
    return _source_positions[index];
  }

  /**
   * @brief Get the weight of a photon source.
   *
   * @param index Index of the photon source, must be in between 0 and
   * get_number_of_sources().
   * @return Weight of the photon source, used to determine how many photons are
   * emitted from this particular source.
   */
  virtual double get_weight(photonsourcenumber_t index) const {
    return _source_luminosities[index] / get_total_luminosity();
  }

  /**
   * @brief Get the total luminosity of all sources together.
   *
   * @return Total luminosity (in s^-1).
   */


  virtual double get_total_luminosity() const {
    double tot_lum = 0.0;
    for (uint_fast32_t i=0;i<_source_luminosities.size();++i) {
      tot_lum += _source_luminosities[i];
    }
    return tot_lum;
  }

  virtual void set_initial_velocity(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator, double timestep) {

    // mgb 16.10.2025
    std::cout << "Source positions size =  " << _source_positions.size() << std::endl;
    std::cout << "Source lifetimes size = " << _source_lifetimes.size() << std::endl;

  //  CoordinateVector<> cell_vel = cell.get_hydro_variables().get_primitives_velocity();
    //_source_velocities[0] = cell_vel;

    for (size_t i=0;i<_source_lifetimes.size();i++){ // mgb want this 
    //while (i < _source_lifetimes.size()) {
      std::cout << "i = " << i << std::endl;
      //float sources
      HydroDensitySubGrid subgrid = *grid_creator->get_subgrid(_source_positions[i]);
      std::cout << "subgrid defined" << std::endl;
      auto cell = subgrid.get_hydro_cell(_source_positions[i]);
      std::cout << "cell defined" << std::endl;
      CoordinateVector<> cell_vel = cell.get_hydro_variables().get_primitives_velocity();
      std::cout<<"Cell velocity defined" << std::endl;
      _source_velocities.push_back(cell_vel);
      std::cout<< "Initial vel defined" << std::endl;
    }
  }

  // if (_moving_sources_flag == true) { // mgb 16.10.2025 cant use if statements here
  virtual void float_sources(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator, double timestep) {

    // mgb 16.10.2025
    std::cout << "Source positions size =  " << _source_positions.size() << std::endl;
    std::cout << "Source lifetimes size = " << _source_lifetimes.size() << std::endl;

  //  CoordinateVector<> cell_vel = cell.get_hydro_variables().get_primitives_velocity();
    //_source_velocities[0] = cell_vel;

    for (size_t i=0;i<_source_lifetimes.size();i++){ // mgb want this 
    //while (i < _source_lifetimes.size()) {
      std::cout << "i = " << i << std::endl;
      //float sources
      HydroDensitySubGrid subgrid = *grid_creator->get_subgrid(_source_positions[i]);
      std::cout << "subgrid defined" << std::endl;
      auto cell = subgrid.get_hydro_cell(_source_positions[i]);
      std::cout << "cell defined" << std::endl;
      CoordinateVector<double> accel = cell.get_hydro_variables().get_gravitational_acceleration();
      std::cout << "Accel defined "<< std::endl;
      _source_velocities[i] += accel*timestep;
      std::cout << "Source velocity has been updated" << std::endl;
       // std::cout << "Source position before = " << _source_positions[i] << std::endl;

      _source_positions[i] = _source_positions[i] + _source_velocities[i]*timestep;
//      std::cout << "Source position after = " << _source_positions[i] << std::endl;

      if (_output_file != nullptr) {
        const CoordinateVector<> &pos = _source_positions[i];
        *_output_file << _total_time << "\t" << pos.x() << "\t" << pos.y()
                      << "\t" << pos.z() << "\t1\t"
                      << _source_indices[i] << "\t" 
                      << _source_luminosities[i] << "\t"
                      << _source_masses[i] << "\t"
                    << "OSTAR moved \n";

    }

    //  << _total_time << "\t" << pos.x() << "\t" << pos.y()
      //                  << "\t" << pos.z() << "\t1\t"
        //                << _source_indices.back() << "\t"
          //              << _source_luminosities.back() << "\t"
            //            << "HOLMES\n";
     }
  }
  //}

    double get_photon_frequency(RandomGenerator &random_generator,
    photonsourcenumber_t index) {

      return _all_spectra[_spectrum_index[index]]->get_random_frequency(random_generator,0.0);

}


  /**
   * @brief Update the distribution after the system moved to the given time.
   *
   * @param simulation_time Current simulation time (in s).
   * @return True if the distribution changed, false otherwise.
   */
   virtual bool update(DensitySubGridCreator< HydroDensitySubGrid > *grid_creator, double actual_timestep) override {

    _total_time += actual_timestep;


    bool updated = false;



  // Type II supernovae: ///

    // clear out sources which no longer exist and add them to SNe todo list 
    // this is where we want to add stellarwinds(mass, temp, lum) // mgb note 11.02.26
    size_t i = 0;
    while (i < _source_lifetimes.size()) {
      _source_lifetimes[i] -= actual_timestep;
      if (_source_lifetimes[i] <= 0.) { // Star Dying so injects supernovae after erasing O star: stellar feedback would copy this but happen every timestep
        // remove the element
        if (_output_file != nullptr) {
          const CoordinateVector<> &pos = _source_positions[i];
          *_output_file << _total_time << "\t" << pos.x() << "\t" << pos.y()
                      << "\t" << pos.z() << "\t1\t"
                      << _source_indices[i] << "\t" 
                      << _source_luminosities[i] << "\t"
                      << _source_masses[i] << "\t"
                      << "SNe II" << "\n";
                    
                    
        //  *_output_file << _total_time << "\t0.\t0.\t0.\t2\t"
          //              << _source_indices[i] << "\t0\t0\tSNe\n";    
        }
        _to_do_feedback.push_back(_source_positions[i]);
        _source_positions.erase(_source_positions.begin() + i);
        _source_lifetimes.erase(_source_lifetimes.begin() + i);
        _source_luminosities.erase(_source_luminosities.begin() + i);
        _spectrum_index.erase(_spectrum_index.begin() + i);
        _source_indices.erase(_source_indices.begin() + i);
        _source_masses.erase(_source_masses.begin() + i);
        _num_sne = _num_sne + 1;
        updated = true;



      } else {
        // check the next element
        ++i;
      }
    }
///   Type Ia supernovae implementation - up to 325 pc scale height
    double _anchor_x = grid_creator->get_box().get_anchor()[0];
    double _anchor_y = grid_creator->get_box().get_anchor()[1];

    double _sides_x = grid_creator->get_box().get_sides()[0];
    double _sides_y = grid_creator->get_box().get_sides()[1];


    double area_kpc = _sides_x*_sides_y/(unit_kpc*unit_kpc);

    std::cout<<"Area = " << area_kpc << std::endl;


    int should_have_done = int(4*area_kpc*_total_time/unit_Myr); //3.15576e13); // taken out *0 mgb 09.10.2025 

    int do_type1 = should_have_done-type1done;

    if (_type1_flag == true) {
    //get simulation box limits

      if (do_type1 > 0) {

        for (int i=0;i<do_type1;i++) {
          double x =
          _anchor_x + _random_generator.get_uniform_random_double() * _sides_x;
          double y =
          _anchor_y + _random_generator.get_uniform_random_double() * _sides_y;

          double z =
            (_type1_sne_scale_height) * // mgb edit 10.11.2025 - added the asymmetric offset term incase using asymmetric box setup
              std::sqrt(-2. *
                        std::log(_random_generator.get_uniform_random_double())) *
              std::cos(2. * M_PI *
                        _random_generator.get_uniform_random_double());
          if (grid_creator->get_box().inside(CoordinateVector<double>(x,y,z))) {
                  _to_do_feedback.push_back(CoordinateVector<double>(x,y,z));
          }
          if (_output_file != nullptr) {
          *_output_file << _total_time << "\t" << x<< "\t" << y
                      << "\t" << z << "\t1\t" 
                      << 0 << "\t" // index
                      << 0 << "\t" // luminosity
                      << 0 << "\t" // mass
                      << "SNe Ia" << "\n";
                    
                    
        //  *_output_file << _total_time << "\t0.\t0.\t0.\t2\t"
          //              << _source_indices[i] << "\t0\t0\tSNe\n";    
        }

          //dotype1
          type1done += 1;
        }

      }
    }



    if ((_total_time > _holmes_time) && (!_holmes_added)) {

      for (uint_fast32_t i=0; i<_number_of_holmes; ++i) {

        double x =
         _anchor_x + _random_generator.get_uniform_random_double() * _sides_x;
        double y =
         _anchor_y + _random_generator.get_uniform_random_double() * _sides_y;
     // we use the Box-Muller method to sample the Gaussian
        double z =
         (_holmes_sh) *
             std::sqrt(-2. *
                       std::log(_random_generator.get_uniform_random_double())) *
             std::cos(2. * M_PI *
                      _random_generator.get_uniform_random_double());
        if (std::abs(z) >= grid_creator->get_box().get_sides()[2]/2.) {
          continue;
        }

        _source_positions.push_back(CoordinateVector<double>(x,y,z));

        double lifetime = 1e99;

        _source_lifetimes.push_back(lifetime);
        _source_luminosities.push_back(_holmes_lum);
        if (_output_file != nullptr) {
          _source_indices.push_back(_next_index);
          ++_next_index;
          const CoordinateVector<> &pos = _source_positions.back();
          *_output_file << _total_time << "\t" << pos.x() << "\t" << pos.y()
                        << "\t" << pos.z() << "\t1\t"
                        << _source_indices.back() << "\t"
                        << _source_luminosities.back() << "\t"
                        << "HOLMES\n";
        }

      }
      _holmes_added = true;
      updated = true;
    }

// Stellar Sources
    if (_total_time - _last_sf > _update_interval) {

      // form cumulative mass structure

      size_t total_cells = grid_creator->number_of_cells();

      std::vector<double> cumulative_mass(total_cells);

      AtomicValue< size_t > igrid(0);
      i = 0;
      double running_mass = 0.0;
      double running_mass_unscaled_for_sfr = 0.0;
      while (igrid.value() < grid_creator->number_of_original_subgrids()) {
        const size_t this_igrid = igrid.post_increment();
        if (this_igrid < grid_creator->number_of_original_subgrids()) {
          HydroDensitySubGrid &subgrid = *grid_creator->get_subgrid(this_igrid);
          for (auto it = subgrid.hydro_begin(); it != subgrid.hydro_end();
               ++it) {

            double cell_mass = it.get_hydro_variables().get_conserved_mass();
            double cell_z = std::abs(it.get_cell_midpoint()[2]); //mgb edit 11.11.2025

            if (cell_z > _scale_height_peak) { //(_scale_height_peak+_asymmetric_offset) || cell_z < (_asymmetric_offset-_scale_height_peak)) { // asymmetric offset mgb 11.10.2025 6.171e18
              cell_mass = 0.0;
            }

            running_mass_unscaled_for_sfr += cell_mass;

            running_mass+= pow(cell_mass, _peakiness_fraction); // pow(1.4) mgb edit add in 11.10.2025 some peak fraction with peak_fraction = 1
            cumulative_mass[i] = running_mass;

            i += 1;


          }
        }
      }
      if (_number_of_updates == 1) {
          if (_restart_flag == true) { // mgb edit 27.01.2026
            init_running_mass = _M_init;
            init_running_mass_unscaled_for_sfr = _M_init_unscaled_for_sfr;
          } else{
              init_running_mass = running_mass;
              init_running_mass_unscaled_for_sfr = running_mass_unscaled_for_sfr;
          }
        //  init_running_mass = running_mass;
         // init_running_mass_unscaled_for_sfr = running_mass_unscaled_for_sfr;
      }

      std::cout << "SFR Ratio from init =  " << running_mass_unscaled_for_sfr/init_running_mass_unscaled_for_sfr << std::endl;
      std::cout << "actual timestep = " << actual_timestep << std::endl;

      for (size_t i=0;i<total_cells;i++) {

        cumulative_mass[i] = cumulative_mass[i]/running_mass;
      }

    double _bursty_star_formation_rate;

    
     double _total_time_for_sfr_burst;
     double _star_formation_rate_area = _star_formation_rate * area_kpc; // need to convert from Myr^-1 kpc^-2 to Myr^-1 mgb edit 11.11.2025 
     if (_restart_flag == true) {
        _total_time_for_sfr_burst = _total_time + _restart_time_myr;
     }
     else {
        _total_time_for_sfr_burst = _total_time;
     }
     if (_bursty_sfr_flag == true) {
      _bursty_star_formation_rate = _star_formation_rate_area + (_burst_amplitude*_star_formation_rate_area - _star_formation_rate_area) * exp(-pow(((_total_time_for_sfr_burst - _time_of_burst_peak)/_length_of_burst), 2)); // edit mgb 19.09.2025 should be in units kg s^-1 kpc^-2
      std::cout << "SFR Amplitude = " << _bursty_star_formation_rate << _bursty_star_formation_rate*unit_Myr/unit_Msol << std::endl;
      std::cout << "Burst amplitude = " << _burst_amplitude << std::endl;
      std::cout << "Star formation rate = " << _star_formation_rate << std::endl;
    } else {
      _bursty_star_formation_rate = _star_formation_rate_area;
     }

      // 0.073 (now 0.207) factor is to take into account we only form stars over 8Msol
      // mass_to_generate in units of Msol to match IMF
      double mass_to_generate = _update_interval*_bursty_star_formation_rate/unit_Msol*0.207*(std::pow(running_mass_unscaled_for_sfr/init_running_mass_unscaled_for_sfr, _kennicutt_schmidt_index)); // replaced 1.988e30 with unit_Msol mgb edit 12.11.2025 

      if (_output_file2 != nullptr) {
        double totallum = get_total_luminosity();
        *_output_file2 << _total_time << "\t" << _total_time_for_sfr_burst << "\t" << totallum << "\t" << _num_sne << "\t" << ((_bursty_star_formation_rate*unit_Myr)/(unit_Msol)) << "\t" << ((_bursty_star_formation_rate*unit_Myr)/(area_kpc*unit_Msol))*std::pow(running_mass_unscaled_for_sfr/init_running_mass_unscaled_for_sfr, _kennicutt_schmidt_index) << "\t" <<  ((_bursty_star_formation_rate/area_kpc)*std::pow(running_mass_unscaled_for_sfr/init_running_mass_unscaled_for_sfr, _kennicutt_schmidt_index)) << "\t" << init_running_mass << "\t" << init_running_mass_unscaled_for_sfr << "\n"; // output the SFR in Msol Myr^-1 kpc^-2 and in kg s^-1 kpc^-2
        _output_file2->flush();

      }

       std::cout << "SHOULD BE GENERATING " << mass_to_generate - _excess_mass<< std::endl;
       std::cout << "Scale height = " << _scale_height << std::endl;
      double mass_generated = 0.0;
      while (mass_generated < mass_to_generate - _excess_mass){
        double m_cur = get_single_mass(_mass_range,_cum_imf,
               _random_generator.get_uniform_random_double());
          double use_density = _random_generator.get_uniform_random_double();
          if(use_density < _peak_fraction) { // I think this is where Peak driving is being used ... mgb 10.10.2025

          double source_pos_val = _random_generator.get_uniform_random_double();
          CoordinateVector<> cell_midpoint;
          double cell_length = 0;
          double cell_z = 0;

          AtomicValue< size_t > igrid(0);
          uint_fast32_t i = 0;

          while (igrid.value() < grid_creator->number_of_original_subgrids()) {
            const size_t this_igrid = igrid.post_increment();
            if (this_igrid < grid_creator->number_of_original_subgrids()) {
              HydroDensitySubGrid &subgrid = *grid_creator->get_subgrid(this_igrid);
              for (auto it = subgrid.hydro_begin(); it != subgrid.hydro_end();
                   ++it) {

                if (cumulative_mass[i] >= source_pos_val){

                  cell_midpoint = it.get_cell_midpoint();
                  cell_length = std::pow(it.get_volume(),1./3.);
                  cell_z = std::abs(it.get_cell_midpoint().z()); //mgb edit 11.11.2025
                 // cell_ionzation = it.get_ionization_variables().get_ionic_fraction(ION_H_n);
                  if (it.get_ionization_variables().get_ionic_fraction(ION_H_n) > 0.5 && it.get_ionization_variables().get_temperature() < 1.1e4 && cell_z < _scale_height ) { // mgb edit 05.03.2026: only form stars in cells which are more than 50% neutral - this is to stop forming stars in already ionised regions 
                    goto afterloop;
                  }


                }

                i += 1;

              }
            }
          }

         afterloop:

        CoordinateVector<> blur;
        blur[0] = _random_generator.get_uniform_random_double()*cell_length - (0.5*cell_length);
        blur[1] = _random_generator.get_uniform_random_double()*cell_length - (0.5*cell_length);
        blur[2] = _random_generator.get_uniform_random_double()*cell_length - (0.5*cell_length);

        _source_positions.push_back(cell_midpoint + blur);

      } else { // And this must be the random driving ... mgb 10.10.2025 
        double x =
         _anchor_x + _random_generator.get_uniform_random_double() * _sides_x;
        double y =
         _anchor_y + _random_generator.get_uniform_random_double() * _sides_y;
     // we use the Box-Muller method to sample the Gaussian
        double z =
         (_scale_height) *
             std::sqrt(-2. *
                       std::log(_random_generator.get_uniform_random_double())) *
             std::cos(2. * M_PI *
                      _random_generator.get_uniform_random_double());

        _source_positions.push_back(CoordinateVector<double>(x,y,z));

      }
        double a0z = 9.955209529401348;
        double a1z = -3.3370109454102326;
        double a2z = 0.8116654874025604;
       // double lifetime = 1.e10 * std::pow(m_cur,-2.5) * 3.154e+7;
       double lifetime = a0z + a1z*std::log10(m_cur) + a2z*(std::log10(m_cur)*std::log10(m_cur));
       lifetime = std::pow(10.0,lifetime);
       lifetime = lifetime*3.154e+7;
        double offset =
              _random_generator.get_uniform_random_double() * _update_interval;
        _source_lifetimes.push_back(lifetime-offset);
        _source_luminosities.push_back(lum_from_mass(m_cur));
        _source_indices.push_back(_next_index);
        _source_masses.push_back(m_cur);
        ++_next_index;
        double interpolatedTemp = interpolate(m_cur, stellarMasses, temperatures);
        size_t closestIndex = findClosestIndex(interpolatedTemp, avail_temps);
        _spectrum_index.push_back(closestIndex);
        std::cout << "MAKING STAR OF MASS " << m_cur << " temp = " << interpolatedTemp << " specindex = " << closestIndex <<  std::endl;
        if (_output_file != nullptr) {
          const CoordinateVector<> &pos = _source_positions.back();
          *_output_file << _total_time << "\t" << pos.x() << "\t" << pos.y()
                        << "\t" << pos.z() << "\t1\t"
                        << _source_indices.back() << "\t"
                        << _source_luminosities.back() << "\t"
                        << m_cur << "\t"
                        << "OSTAR\n";
        }

        mass_generated += m_cur;
      }
      if (mass_generated == 0) {
        _excess_mass = _excess_mass - mass_to_generate;
        std::cout << "Still over mass to generate, decreased excess to " << _excess_mass << std::endl;

      } else {
        _excess_mass = mass_generated - mass_to_generate + _excess_mass;
        std::cout << "OVER SHOT, saving excess of" << _excess_mass << std::endl;

      }

        _last_sf = _total_time;
        updated = true;
        ++_number_of_updates;
    }


      if (_output_file != nullptr) {
        _output_file->flush();
      }

    return updated;
  }


// --------------------------------------

  /**
   * @brief Write the distribution to the given restart file.
   *
   * @param restart_writer RestartWriter to use.
   */
  virtual void write_restart_file(RestartWriter &restart_writer) const {

    restart_writer.write(_star_formation_rate);
    restart_writer.write(_bursty_sfr_flag);
    restart_writer.write(_length_of_burst); // mgb edit 19.09.2025
    restart_writer.write(_time_of_burst_peak); // mgb edit 19.09.2025
    restart_writer.write(_burst_amplitude); // mgb edit 19.09.2025
    restart_writer.write(_peakiness_fraction); // mgb edit 11.11.2025
    restart_writer.write(_scale_height_peak); // mgb edit 11.11.2025 
    restart_writer.write(_type1_sne_scale_height); // mgb edit 11.11.2025
    restart_writer.write(_type1_flag); // mgb edit 12.11.2025
    restart_writer.write(_kennicutt_schmidt_index); // mgb edit 12.11.2025
    restart_writer.write(_restart_flag); // mgb edit 21.11.2025
    restart_writer.write(_restart_time_myr);
    restart_writer.write(_M_init); // mgb edit 27.01.2026
    restart_writer.write(_M_init_unscaled_for_sfr); // mgb edit 27.01.2026
    //restart_writer.write(_moving_sources_flag);
    restart_writer.write(_update_interval);
    restart_writer.write(_lum_adjust);
    restart_writer.write(_excess_mass);
    restart_writer.write(_scale_height);
    restart_writer.write(_peak_fraction);
    restart_writer.write(init_running_mass);
    restart_writer.write(_num_sne);
    restart_writer.write(_holmes_time);
    restart_writer.write(_holmes_sh);
    restart_writer.write(_holmes_lum);
    restart_writer.write(_number_of_holmes);
    restart_writer.write(type1done);
    restart_writer.write(_total_time);
    restart_writer.write(_holmes_added); 
    restart_writer.write(_last_sf);
    _random_generator.write_restart_file(restart_writer);
    {
      const auto size = _source_positions.size();
      restart_writer.write(size);
      for (std::vector< CoordinateVector<> >::size_type i = 0; i < size; ++i) {
        _source_positions[i].write_restart_file(restart_writer);
      }
    }
    {
      const auto size = _source_lifetimes.size();
      restart_writer.write(size);
      for (std::vector< double >::size_type i = 0; i < size; ++i) {
        restart_writer.write(_source_lifetimes[i]);
      }
    }
    {
      const auto size = _source_luminosities.size();
      restart_writer.write(size);
      for (std::vector< double >::size_type i = 0; i < size; ++i) {
        restart_writer.write(_source_luminosities[i]);
      }

    }
    {
      const auto size = _source_masses.size();
      restart_writer.write(size);
      for (std::vector< double >::size_type i = 0; i < size; ++i) {
        restart_writer.write(_source_masses[i]);
      }
    }
    restart_writer.write(_number_of_updates);
    const bool has_output = (_output_file != nullptr);
    restart_writer.write(has_output);
    if (has_output) {
      // store current position in the std::ofstream
      // we want to be able to continue writing from that point
      const auto filepos = _output_file->tellp();
      restart_writer.write(filepos);
      const auto filepos2 = _output_file2->tellp();
      restart_writer.write(filepos2);
      {
        const auto size = _source_indices.size();
        restart_writer.write(size);
        for (std::vector< uint_fast32_t >::size_type i = 0; i < size; ++i) {
          restart_writer.write(_source_indices[i]);
        }
      }
      restart_writer.write(_next_index);
    }
  }

  /**
   * @brief Restart constructor.
   *
   * @param restart_reader Restart file to read from.
   */
  inline BurstyPhotonSourceDistribution(RestartReader &restart_reader)
      : _star_formation_rate(restart_reader.read< double >()),
        _bursty_sfr_flag(restart_reader.read< bool >()),
      //  _moving_sources_flag(restart_reader.read< bool >())
        _length_of_burst(restart_reader.read< double >()), // mgb edit 19.09.2025
        _time_of_burst_peak(restart_reader.read< double >()), // mgb edit 19.09.2025
        _burst_amplitude(restart_reader.read< double >()), // mgb edit 19.09.2025
        _peakiness_fraction(restart_reader.read< double >()), // mgb edit 11.11.2025
        _scale_height_peak(restart_reader.read< double >()), // mgb edit 11.11.2025
        _type1_sne_scale_height(restart_reader.read< double >()), // mgb edit 11.11.2025
        _type1_flag(restart_reader.read< bool >()), // mgb edit 12.11.2025
        _kennicutt_schmidt_index(restart_reader.read< double>()), // mgb edit 12.11.2025
        _restart_flag(restart_reader.read< double >()), // mgb edit 21.11.2025
        _restart_time_myr(restart_reader.read< double >()), // mgb edit 08.12.2025 
        _M_init(restart_reader.read< double >()), // mgb edit 27.01.2026
        _M_init_unscaled_for_sfr(restart_reader.read< double >()), // mgb edit 27.01.2026
        _update_interval(restart_reader.read< double >()),
        _lum_adjust(restart_reader.read< double >()),
        _excess_mass(restart_reader.read<double>()),
        _scale_height(restart_reader.read<double>()),
        _peak_fraction(restart_reader.read<double>()),
        init_running_mass(restart_reader.read<double>()),
        _num_sne(restart_reader.read<double>()),
        _holmes_time(restart_reader.read<double>()),
        _holmes_sh(restart_reader.read<double>()),
        _holmes_lum(restart_reader.read<double>()),
        _number_of_holmes(restart_reader.read<uint_fast32_t>()),
        type1done(restart_reader.read<int>()),
        _total_time(restart_reader.read<double>()),
        _holmes_added(restart_reader.read<bool>()),
        _last_sf(restart_reader.read<double>()),
        _random_generator(restart_reader) {

    {
      const std::vector< CoordinateVector<> >::size_type size =
          restart_reader.read< std::vector< CoordinateVector<> >::size_type >();
      _source_positions.resize(size);
      for (std::vector< CoordinateVector<> >::size_type i = 0; i < size; ++i) {
        _source_positions[i] = CoordinateVector<>(restart_reader);
      }
    }
    {
      const std::vector< double >::size_type size =
          restart_reader.read< std::vector< double >::size_type >();
      _source_lifetimes.resize(size);
      for (std::vector< double >::size_type i = 0; i < size; ++i) {
        _source_lifetimes[i] = restart_reader.read< double >();
      }
    }
    {
      const std::vector< double >::size_type size =
          restart_reader.read< std::vector< double >::size_type >();
      _source_luminosities.resize(size);
      for (std::vector< double >::size_type i = 0; i < size; ++i) {
        _source_luminosities[i] = restart_reader.read< double >();
      }
    }
    {
      const std::vector< double >::size_type size =
          restart_reader.read< std::vector< double >::size_type >();
      _source_masses.resize(size);
      for (std::vector< double >::size_type i = 0; i < size; ++i) {
        _source_masses[i] = restart_reader.read< double >();
      }
    }
    _number_of_updates = restart_reader.read< uint_fast32_t >();
    const bool has_output = restart_reader.read< bool >();
    if (has_output) {
      const std::streampos filepos = restart_reader.read< std::streampos >();
      // truncate the original file to the size we were at
      if (truncate("Bursty_source_positions.txt", filepos) != 0) {
        cmac_error("Error while truncating output file!");
      }
      // now open the file in append mode
      _output_file = new std::ofstream("Bursty_source_positions.txt",
                                       std::ios_base::app);

      const std::streampos filepos2 = restart_reader.read< std::streampos >();
                                       // truncate the original file to the size we were at
      if (truncate("TotalLuminosity.txt", filepos2) != 0) {
              cmac_error("Error while truncating output file!");
            }
                                       // now open the file in append mode
      _output_file2 = new std::ofstream("TotalLuminosity.txt",
                                            std::ios_base::app);


      {
        const std::vector< uint_fast32_t >::size_type size =
            restart_reader.read< std::vector< uint_fast32_t >::size_type >();
        _source_indices.resize(size);
        for (std::vector< uint_fast32_t >::size_type i = 0; i < size; ++i) {
          _source_indices[i] = restart_reader.read< uint_fast32_t >();
        }
      }
      _next_index = restart_reader.read< uint_fast32_t >();
    }


        // form cumulative IMF
        double imf_start = 8.0;
        double imf_end = 120;

        double full_area = integral(kroupa_imf, imf_start, imf_end, 10000);


        uint_fast32_t range_length = 10000;
        for (uint_fast32_t i=0; i< range_length; ++i){
          double step = (imf_end-imf_start)/range_length;
          _mass_range.push_back(imf_start + step*i);
          double part_integral = integral(kroupa_imf, imf_start,imf_start + step*i,10000);
          _cum_imf.push_back(part_integral/full_area);
        }

  novahandler = new SupernovaHandler(_sne_energy);
  }
};

#endif // BURSTYPHOTONSOURCEDISTRIBUTION_HPP
