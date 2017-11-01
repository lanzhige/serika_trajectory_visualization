#include "geoArray.h"
#include "selector.h"

#include <thrust/copy.h>
#include <thrust/find.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

namespace selika{
GeoArray::GeoArray(){}

int GeoArray::add(int id, double cnt_lat, double cnt_lng, double radius
    , long start_time, long end_time){
  id_array.push_back(id);
  cnt_lat_array.push_back(cnt_lat);
  cnt_lng_array.push_back(cnt_lng);
  radius_array.push_back(radius);
  start_time_array.push_back(start_time);
  end_time_array.push_back(end_time);
  return 0;
}

int GeoArray::clear(){
  id_array.clear();
  cnt_lat_array.clear();
  cnt_lng_array.clear();
  radius_array.clear();
  start_time_array.clear();
  end_time_array.clear();
  id_geo_index.clear();
  chord_list.clear();
  return 0;
}

int GeoArray::remove(int id){
  thrust::device_vector<int>::iterator iter;
  iter = thrust::find(id_array.begin(), id_array.end(), id);
  if (iter == id_array.end()){
    printf("Can't find exact obj with id %d to remove!\n",id);
    return 1;
  }
  int len = iter - id_array.begin();
  id_array.erase(iter);
  cnt_lat_array.erase(cnt_lat_array.begin()+len);
  cnt_lng_array.erase(cnt_lng_array.begin()+len);
  radius_array.erase(radius_array.begin()+len);
  start_time_array.erase(start_time_array.begin()+len);
  end_time_array.erase(end_time_array.begin()+len);
  return 0;
}

int GeoArray::selectData(comdb &db, int id){
//*currently just support select trajectory in one circle area
  id_geo_index.clear();
  thrust::device_vector<int>::iterator iter;
  iter = thrust::find(id_array.begin(),id_array.end(),id);
//find the index of selected area.
  if (iter == id_array.end()){
    printf("Can't find exact obj with id %d to select data!\n",id);
    return 1;
  }
  int len = iter - id_array.begin();
  iter = thrust::max_element(db.col_id_vec.begin(), db.col_id_vec.end());
  int id_len = iter - db.col_id_vec.begin();
  thrust::device_vector<int> traj_id_array(id_len, -1);
  //*id_len should be the value at db.col_id_vec[id_len]
  Selector selector_(
      thrust::raw_pointer_cast(traj_id_array.data())
      , thrust::raw_pointer_cast(db.col_id_vec.data())
      , thrust::raw_pointer_cast(db.col_time_vec.data())
      , thrust::raw_pointer_cast(db.col_lat_vec.data())
      , thrust::raw_pointer_cast(db.col_lon_vec.data())
      , thrust::raw_pointer_cast(start_time_array.data())+len
      , thrust::raw_pointer_cast(end_time_array.data())+len
      , thrust::raw_pointer_cast(cnt_lat_array.data())+len
      , thrust::raw_pointer_cast(cnt_lng_array.data())+len
      , thrust::raw_pointer_cast(radius_array.data())+len
  );
  thrust::counting_iterator<int> begin(0);
  thrust::counting_iterator<int> end(db.col_id_vec.size());
  thrust::for_each(begin, end, selector_);
  //ids of trajectory which go through the selected area
  len = id_array.size();
  thrust::device_vector<int> index(db.col_id_vec.size()*len);
  //generate a table of trajectory points * geographical object areas
  in_range in_range_(
      id_len, thrust::raw_pointer_cast(traj_id_array.data())
      , thrust::raw_pointer_cast(db.col_time_vec.data())
      , thrust::raw_pointer_cast(db.col_lat_vec.data())
      , thrust::raw_pointer_cast(db.col_lon_vec.data())
      , len
      , thrust::raw_pointer_cast(start_time_array.data())
      , thrust::raw_pointer_cast(end_time_array.data())
      , thrust::raw_pointer_cast(cnt_lat_array.data())
      , thrust::raw_pointer_cast(cnt_lng_array.data())
      , thrust::raw_pointer_cast(radius_array.data())
  );
  thrust::sequence(index.begin(), index.end());
  id_geo_index.resize(db.col_id_vec.size()*len, -1);
  thrust::copy_if(
      index.begin(), index.end(), id_geo_index.begin(), in_range_
  );
  iter = thrust::find(id_geo_index.begin(), id_geo_index.end(), -1);
  id_geo_len = iter - id_geo_index.begin();
  id_geo_index.resize(id_geo_len);
  return 0;
}

int GeoArray::genChord(int geo_id) {
  chord_len = id_geo_len;
  int *temp = new int[chord_len];
  thrust::copy_n(id_geo_index.begin(), chord_len, temp);
  int len = id_array.size();
  chord = new int[(len-1)*(len-1)];
  for (int i=0;i<(len-1)*(len-1);i++) chord[i]=0;
  for (int i=0;i<chord_len;i++)
    if (temp[i]%len ==geo_id){
      if (chord_len-1>i>0&&(temp[i]/len == temp[i-1]/len)){
        int from = temp[i-1]%len;
        int to = temp[i+1]%len;
        chord[from*(len-1)+to]++;
      }
    }
  chord_len = (len-1)*(len-1);
  return 0;
}
}  //  selika
