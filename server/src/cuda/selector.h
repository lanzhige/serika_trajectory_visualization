#ifndef SELIKA_SELECTOR_H_
#define SELIKA_SELECTOR_H_

#include <thrust/device_vector.h>

#include "comdb.h"
#include "lexicographical_sort.h"

namespace selika{
#ifndef MAX_INT
#define MAX_INT 2147483647
#endif

#ifndef PI
#define PI 3.1415926535
#endif

#ifndef GLOBAL_R
#define GLOBAL_R 6371e3
#endif

struct GlobalData {
//structure to sort points in database based on index id, time
  comdb *db;
  GlobalData(comdb *db);
  void sortByIdTime();
};

struct Selector{
  //structure to judge if a point is in the circle area
  //*note: this operator() function should change to the same as operator() in
  // in_range structure to judge if an edge cross through the circle
  Selector();
  Selector(int *id_array, int *db_id, long *db_time
      , double *db_lat, double *db_lng
      , long *start_time, long *end_time, double *cnt_lat
      , double *cnt_lng, double *radius);

  int db_len;
  long *start_time, *end_time;
  double *cnt_lat, *cnt_lng, *radius;
  
  int *db_id;
  int *id_array;
  long *db_time;
  double *db_lat, *db_lng;

  int id_len;

  //haversine function to calculate meters between latLng coordinates
  __host__ __device__
  double haversine(double lat, double lng, double cnt_lat, double cnt_lng){
    double delta_lat = (lat - cnt_lat)*PI/180.0;
    double delta_lng = (lng - cnt_lng)*PI/180.0;
    double lat1 = cnt_lat*PI/180.0;
    double lat2 = lat*PI/180.0;
    double a = sinf(delta_lat/2.0) *sinf(delta_lat/2.0)
        +cosf(lat1) *cosf(lat2) *sinf(delta_lng/2.0) *sinf(delta_lng/2.0);
    double c = atan2f(sqrtf(a),sqrtf(1.0-a));
    return GLOBAL_R*c;
  }

  template <typename T>
  __host__ __device__
  void operator()(const T &i){
    double lat = db_lat[i], lng = db_lng[i];
    double c_lat = *cnt_lat, c_lng = *cnt_lng;
    if (*start_time<=db_time[i] &&*end_time>db_time[i]
        && *radius>=haversine(lat, lng, c_lat, c_lng)) {
      id_array[db_id[i]] = db_id[i];
    };
  }
};

struct in_range{
//structure to judge if an edge passes through a circle area, if so return true
  int len, traj_id_len;

  int *traj_id_array;
  long *db_time, *start_time_array, *end_time_array;
  double *db_lat, *db_lng, *cnt_lat_array, *cnt_lng_array, *radius_array;

  __host__ __device__
  double haversine(double lat, double lng, double *cnt_lat, double *cnt_lng){
    double delta_lat = (lat - *cnt_lat)*PI/180.0;
    double delta_lng = (lng - *cnt_lng)*PI/180.0;
    double lat1 = *cnt_lat*PI/180.0;
    double lat2 = lat*PI/180.0;
    double a = sinf(delta_lat/2.0) *sinf(delta_lat/2.0)
        +cosf(lat1) *cosf(lat2) *sinf(delta_lng/2.0) *sinf(delta_lng/2.0);
    double c = 2*atan2f(sqrtf(a),sqrtf(1.0-a));
    return GLOBAL_R*c;
  }

  in_range(int traj_id_len, int *traj_id_array, long *db_time, double *db_lat
      , double *db_lng, int len, long *start_time_array, long *end_time_array
      , double *cnt_lat_array, double *cnt_lng_array, double *radius_array);

  __host__ __device__
  bool operator()(const int &i){
    int geo_index = i%len;
    int traj_index = i/len;
    if (traj_index>=traj_id_len||traj_id_array[traj_index]==-1) return 0;
    if (db_time[traj_index]<start_time_array[geo_index]) return 0;
    if (db_time[traj_index]>end_time_array[geo_index]) return 0;
    if (radius_array[geo_index]>=haversine(db_lat[traj_index]
        , db_lng[traj_index], cnt_lat_array+geo_index
        , cnt_lng_array+geo_index)) return 1;
    if (traj_index>0){
      /*if (-0.0000001<(db_lat[traj_index]-db_lat[traj_index-1])<0.0000001
          && -0.0000001<(db_lng[traj_index]-db_lng[traj_index-1])<0.0000001)
          return 0;*/
      double A = (db_lat[traj_index]-db_lat[traj_index-1])
          /(db_lng[traj_index]-db_lng[traj_index-1]);
      double B = db_lat[traj_index]-A*db_lng[traj_index];
      double m = cnt_lng_array[geo_index] + A*cnt_lat_array[geo_index];
      double lng = (m-A*B)/(A*A+1);
      double lat = (A*lng+B);
      if (lat!=lat || lng!=lng) return 0;
      if (radius_array[geo_index]>=haversine(lat, lng
          , cnt_lat_array+geo_index, cnt_lng_array+geo_index)) return 1;
    }
    return 0;
  }
};

struct passed_geo{
//structure to find if two points can form an edge
  int geo_len, geo_id, id_geo_len;
  int *id_geo_index;
  passed_geo(int geo_len, int *id_geo_index, int geo_id, int id_geo_len);

  __host__ __device__
  bool operator()(const int &i){
    if (id_geo_index[i]%geo_len == geo_id) return 1;
    if (i<id_geo_len-1){
      if (id_geo_index[i+1]%geo_len == geo_id
          &&id_geo_index[i]/geo_len == id_geo_index[i+1]/geo_len) return 1;
    }
    if (0<i){
      if (id_geo_index[i-1]%geo_len == geo_id
          &&id_geo_index[i]/geo_len == id_geo_index[i-1]/geo_len) return 1;    
    }
    return 0;
  }
};
}  //selika
#endif

