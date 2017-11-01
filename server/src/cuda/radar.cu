#include "radar.h"
#include "lexicographical_sort.h"

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/replace.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>
#include <thrust/system/tbb/execution_policy.h>
#include <thrust/sequence.h>

#include <iostream>
namespace selika{
namespace radar{
Radar::Radar(){}

Radar::Radar(long int start_time, long int end_time, long int time_window
    , long int time_interval, double r, double lat, double lng, int arg)
    : start_time(start_time), end_time(end_time), time_window(time_window)
    , time_interval(time_interval), r(r), lat(lat), lng(lng), arg(arg){
    std::vector<double> temp(arg*2);
    div_vec=temp;
  }

int Radar::init(){
  thrust::counting_iterator<unsigned int> begin(0);
  thrust::counting_iterator<unsigned int> end(arg);
  radar_divide divide(lat,lng,r,
      (double *)thrust::raw_pointer_cast(div_vec.data()),arg);
  thrust::for_each(begin,end,divide);

  return 1;
}

int *Radar::query_radar(const comdb &db){
  int slice_num=1+ceil((end_time-start_time-time_window)/time_interval);
  int duplicate_time=ceil(time_window/time_interval);
  int *values = new int[arg*slice_num];
  thrust::device_vector<int> id_vec((db.size-1)*duplicate_time);
  thrust::device_vector<int> cell_vec((db.size-1)*duplicate_time);
  thrust::fill(id_vec.begin(),id_vec.end(),-1);
  thrust::fill(cell_vec.begin(),cell_vec.end(),-1);

  int col_len=db.col_id_vec.size();
  thrust::device_vector<int> col_id_vec(col_len);
  thrust::device_vector<long int> col_time_vec(col_len);
  thrust::device_vector<double> col_lat_vec(col_len);
  thrust::device_vector<double> col_lng_vec(col_len);
  thrust::copy_n(db.col_id_vec.begin(), col_len, col_id_vec.begin());
  thrust::copy_n(db.col_time_vec.begin(), col_len, col_time_vec.begin());
  thrust::copy_n(db.col_lat_vec.begin(), col_len, col_lat_vec.begin());
  thrust::copy_n(db.col_lon_vec.begin(), col_len, col_lng_vec.begin());

/*int *output = new int(10000);
int *timeout = new int (10000);
thrust::copy_n(col_id_vec.begin(),10000,output);
thrust::copy_n(col_time_vec.begin(),10000,timeout);
for (int i=0;i<9999;i++){
  std::cerr<<output[i]<<" "<<timeout[i]<<std::endl;
}
*/

  thrust::device_vector<int> permutation(col_len);
  thrust::sequence(permutation.begin(),permutation.end());

  update_permutation(col_time_vec, permutation);
  update_permutation(col_id_vec,permutation);

  apply_permutation(col_id_vec,permutation);
  apply_permutation(col_time_vec,permutation);
  apply_permutation(col_lat_vec,permutation);
  apply_permutation(col_lng_vec,permutation);

  thrust::counting_iterator<unsigned int> begin(0);
  thrust::counting_iterator<unsigned int> end(db.size-1);
  QueryCell query_cell(start_time,end_time,time_window,time_interval
                , r, lat, lng, arg
                , thrust::raw_pointer_cast(div_vec.data())
                , thrust::raw_pointer_cast(col_lat_vec.data())
                , thrust::raw_pointer_cast(col_lng_vec.data())
                , thrust::raw_pointer_cast(col_time_vec.data())
                , thrust::raw_pointer_cast(col_id_vec.data())
                , thrust::raw_pointer_cast(id_vec.data())
                , thrust::raw_pointer_cast(cell_vec.data()));
  thrust::for_each(begin,end,query_cell);

  thrust::sort_by_key(cell_vec.begin(),cell_vec.end(),id_vec.begin());
  thrust::device_vector<int> count_vec(slice_num*arg);
  thrust::fill(count_vec.begin(),count_vec.end(),0);
  EqualPlus equal_plus(thrust::raw_pointer_cast(cell_vec.data()),
      thrust::raw_pointer_cast(id_vec.data()),
      thrust::raw_pointer_cast(count_vec.data()));

  thrust::counting_iterator<unsigned int> end_seq((db.size-1)*duplicate_time);
  thrust::for_each(begin,end_seq,equal_plus);

  thrust::copy_n(count_vec.begin(),slice_num*arg,values);

  return values;

}
/*
int* Radar::query_radar(const comdb &p_db){
  int *values = new int[arg];
  query_cell qc(arg, (double *)thrust::raw_pointer_cast(div_vec.data()),
      lat,lng,r,start_time, end_time);
  
  thrust::device_vector<int> id_vec(p_db->size);
  thrust::device_vector<int> cell_vec(p_db->size);
  
  thrust::for_each(
      thrust::make_zip_iterator(thrust::make_tuple(
          p_db->col_lon_vec.begin(),p_db->col_lon_vec.begin()+1,
          p_db->col_lat_vec.begin(),p_db->col_lat_vec.begin()+1,
          p_db->col_time_vec.begin(),p_db->col_id_vec.begin(),
          p_db->col_id_vec.begin()+1,id_vec.begin(),cell_vec.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
          p_db->col_lon_vec.end()-1,p_db->col_lon_vec.end(),
          p_db->col_lat_vec.end()-1,p_db->col_lat_vec.end(),
          p_db->col_time_vec.end(),p_db->col_id_vec.end()-1,
          p_db->col_id_vec.end(),id_vec.begin(),cell_vec.end())),
      qc
  );  
  
  thrust::sort_by_key(cell_vec.begin(), cell_vec.end(), id_vec.begin());
  thrust::device_vector<int> count_vec(p_db->size);
  thrust::fill(count_vec.begin(), count_vec.end(), 0);
  equal_plus ep(thrust::raw_pointer_cast(cell_vec.data()),
      thrust::raw_pointer_cast(id_vec.data()),
      thrust::raw_pointer_cast(count_vec.data()));

  thrust::counting_iterator<unsigned int> begin(0);
  thrust::counting_iterator<unsigned int> end(p_db->size);
  thrust::for_each(begin, end, ep);
    
  thrust::copy_n(count_vec.begin(), arg , values);
  thrust::copy_n(count_vec.begin(), arg,
      std::ostream_iterator<int>(std::cout, ","));

  return values;
}*/
}  //radar
}  //selika
