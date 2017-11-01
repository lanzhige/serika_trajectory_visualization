#include "selector.h"

//#include <thrust/system/tbb/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace selika{
GlobalData::GlobalData(comdb *db) :db(db) {}

void GlobalData::sortByIdTime() {
  int len = db->col_id_vec.size();
  //id = new thrust::device_vector<int>(len);
  //time = new thrust::device_vector<long>(len);
  //lat = new thrust::device_vector<double>(len);
  //lng = new thrust::device_vector<double>(len);
  //thrust::copy_n(db->col_id_vec.begin(),len,id->begin());
  //thrust::copy_n(db->col_time_vec.begin(),len,time->begin());
  //thrust::copy_n(db->col_lat_vec.begin(),len,lat->begin());
  //thrust::copy_n(db->col_lon_vec.begin(),len,lng->begin());
//----permutation as index
  thrust::device_vector<int> permutation(len);
  thrust::sequence(permutation.begin(),permutation.end());
//----set dictionary order
  update_permutation(db->col_time_vec,permutation);
  update_permutation(db->col_id_vec,permutation);

  apply_permutation(db->col_id_vec, permutation);
  apply_permutation(db->col_time_vec, permutation);
  apply_permutation(db->col_lat_vec, permutation);
  apply_permutation(db->col_lon_vec, permutation);
  
  return;
}

Selector::Selector(int *id_array, int *db_id, long *db_time, double *db_lat
    , double *db_lng, long *start_time, long *end_time, double *cnt_lat
    , double *cnt_lng, double *radius)
    :id_array(id_array), db_id(db_id), db_time(db_time), db_lat(db_lat)
    , db_lng(db_lng), start_time(start_time), end_time(end_time)
    , cnt_lat(cnt_lat), cnt_lng(cnt_lng), radius(radius) { }

in_range::in_range(int traj_id_len, int *traj_id_array, long *db_time
    , double *db_lat, double *db_lng, int len, long *start_time_array
    , long *end_time_array, double *cnt_lat_array
    , double *cnt_lng_array, double *radius_array)
    :traj_id_len(traj_id_len), traj_id_array(traj_id_array)
    , db_time(db_time), db_lat(db_lat), db_lng(db_lng)
    , len(len), radius_array(radius_array)
    , cnt_lat_array(cnt_lat_array), cnt_lng_array(cnt_lng_array)
    , start_time_array(start_time_array), end_time_array(end_time_array) { }

passed_geo::passed_geo(int geo_len, int *id_geo_index, int geo_id
    , int id_geo_len)
    :geo_len(geo_len), id_geo_index(id_geo_index), geo_id(geo_id)
    , id_geo_len(id_geo_len) { 
}
} // selika
