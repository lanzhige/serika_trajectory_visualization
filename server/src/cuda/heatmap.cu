#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/fill.h>

#include "heatmap.h"

namespace selika{
namespace heatmap{
HeatMap::HeatMap(double lat_top, double lat_bottom, double lng_left
    , double lng_right, int grids_row, int grids_col, long int start_time
    , long int end_time, long int time_window, long int time_interval)
          :lat_top(lat_top),lat_bottom(lat_bottom),lng_left(lng_left)
          ,lng_right(lng_right),grids_row(grids_row),grids_col(grids_col)
          ,start_time(start_time),end_time(end_time)
          ,time_window(time_window) ,time_interval(time_interval){ }

int* HeatMap::query_heatmap(const comdb &db){
  //for each time slice of heatmap build a cell table to store
  int slice_num=1+ceil((end_time-start_time-time_window)/time_interval);
  int duplicate_time=ceil(time_window/time_interval);
  int *values = new int[slice_num*grids_row*grids_col];
  //*Indeed, it doesn't have to lengthen the id array, can be changed here
  thrust::device_vector<int> id_vec(db.size*duplicate_time);
  thrust::device_vector<int> cell_vec(db.size*duplicate_time);
  thrust::fill(cell_vec.begin(),cell_vec.end(),-1);
  thrust::fill(id_vec.begin(),id_vec.end(),-1);

  thrust::counting_iterator<unsigned int> begin(0);
  thrust::counting_iterator<unsigned int> end(db.size);

  QueryCell query_cell(
                lat_top,lat_bottom,lng_left,lng_right,grids_row,grids_col
                ,start_time,end_time,time_window,time_interval
                ,thrust::raw_pointer_cast(db.col_lon_vec.data())
                ,thrust::raw_pointer_cast(db.col_lat_vec.data())
                ,thrust::raw_pointer_cast(db.col_time_vec.data())
                ,thrust::raw_pointer_cast(db.col_id_vec.data())
                ,thrust::raw_pointer_cast(id_vec.data())
                ,thrust::raw_pointer_cast(cell_vec.data()));
  thrust::for_each(begin,end,query_cell);
  thrust::sort_by_key(cell_vec.begin(),cell_vec.end(),id_vec.begin());

  thrust::device_vector<int> count_vec(slice_num*grids_row*grids_col);

  thrust::fill(count_vec.begin(),count_vec.end(),0);

  EqualPlus equal_plus(thrust::raw_pointer_cast(cell_vec.data()),
      thrust::raw_pointer_cast(id_vec.data()),
      thrust::raw_pointer_cast(count_vec.data()));

  thrust::counting_iterator<unsigned int> end_seq(db.size*duplicate_time);
  thrust::for_each(begin,end_seq,equal_plus);

  thrust::copy_n(count_vec.begin(),
      slice_num*grids_row*grids_col,values);

  return values;
}
}  //heatmap
}  //selika
