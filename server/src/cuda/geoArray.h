#ifndef SELIKA_GEOARRAY_H_
#define SELIKA_GEOARRAY_H_

#include "selector.h"

#include <thrust/device_vector.h>

namespace selika{
struct GeoArray{
//storage of geographical object data
  GeoArray();
  thrust::device_vector<int> id_array;
  thrust::device_vector<double> cnt_lat_array, cnt_lng_array, radius_array;
  thrust::device_vector<long> start_time_array, end_time_array;
  thrust::device_vector<int> id_geo_index;
  int id_geo_len;
  thrust::device_vector<int> chord_list;
  int *chord, chord_len;
  int add(int id, double cnt_lat, double cnt_lng, double radius
      , long start_time, long end_time);
  int clear();
  int selectData(comdb &db, int id);
  int genChord(int geo_id);
  int remove(int id);
};
}  //  selika
#endif
