#ifndef COMDB_H_
#define COMDB_H_
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "trajectory.h"

//size_t size;					 // table size
//thrust::device_vector<float>    col_lat_vec;         // store trajectory column's values
//thrust::device_vector<int>    	col_id_vec;          // store id values
//thrust::device_vector<float>    res_lat_vec;     // store query result

struct comdb
{
  //  thrust::device_vector<float>    col_lat_vec;         // store trajectory column's values
   // thrust::device_vector<int>      col_id_vec;          // store id values
    //thrust::device_vector<float>    res_lat_vec;     	 // store query result
    thrust::device_vector<int>   col_id_vec;
    thrust::device_vector<double> col_lat_vec;
    thrust::device_vector<double> col_lon_vec;
    thrust::device_vector<long> col_time_vec; 
    thrust::device_vector<double> res_lat_vec;
    thrust::device_vector<double> res_lon_vec;
    thrust::device_vector<long> res_time_vec;
    thrust::device_vector<int> res_id_vec;


    size_t size;                    // table size

    comdb() 
    {
	col_id_vec = {};
	col_lat_vec = {};
	col_time_vec = {};
	col_lon_vec = {};
	res_lat_vec = {};
	res_id_vec = {};
    }

    /*void init(size_t _size) 
    {
	size = _size;

	thrust::host_vector<float> lat_host(size);
	col_lat_vec = lat_host;
	
	thrust::host_vector<int> id_host(size);
	col_id_vec = id_host;

	printf("..........comdb %d\n", col_id_vec.data());

	thrust::host_vector<float> res_host(size);
	res_lat_vec = res_host;
    }
	*/
    int select_all_id(int *result);    							// get all trajectory ids

    void select_by_id(int id);            		     	     		// query latitude by id

    int select_by_time(const char *start, const char *end);			// query latitude by a time interval
    
	
    int select_by_space(double top_left_lon, double top_left_lat, 
		double bottom_right_lon, double bottom_right_lat);		// query id by a given space window
    int select_by_space_time(double top_left_lon, double top_left_lat,
		double bottom_right_lon, double bottom_right_lat,		// query id by a given space-time window
		long start, long end);

    double* select_range_location();						// query range of longitude and latitude
};

int load_data(const char* filename, comdb &db);  	             // load data from file


struct id_equal
{
    int id;

    __host__ __device__
    id_equal(int _id) : id(_id) { }

    // read each line of id, if id matches return the corresponding column value
    // id, trajectory, and result form a Tuple
    template <typename Tuple>
   __host__  __device__
    void operator() (const Tuple t)
    {
        int data = thrust::get<0>(t);
        double lat = thrust::get<1>(t);
	double lon = thrust::get<2>(t);
	long ts = thrust::get<3>(t);
        
	if (data == id) 
	{
		thrust::get<4>(t) = id;
		//printf("--------%ld\n", ts);
		thrust::get<5>(t) = lat;
		thrust::get<6>(t) = lon;
		thrust::get<7>(t) = ts;
	}
        else   
	{
		thrust::get<4>(t) = -1;
		thrust::get<5>(t) = -1;
		thrust::get<6>(t) = -1;
		thrust::get<7>(t) = 0;
	}

    }
};

struct get_all_id
{
	int *all_id, *in_data;

	get_all_id(int *ids, int *in) : all_id(ids), in_data(in) {}

	__host__ __device__
	void operator() (const int& i)
	{
		if (i == 0) all_id[i] = in_data[i];
		if (i > 0 && in_data[i] > 0 && in_data[i] != in_data[i-1])
		{
			all_id[i] = in_data[i];		
		}
		else 
		{
			all_id[i] = -1;
		}
	}
};

struct time_between
{
	long start, end;
	
	__host__ __device__
	time_between(long s, long e) : start(s), end(e) {}
	
	
	__host__ __device__ 
	int cu_strcmp(const char *str_a, const char *str_b, unsigned int len = 256)
	{
		int match = 0;
		unsigned i = 0;
		unsigned done = 0;

		while ((i < len) && (match == 0) && !done)
		{	
			// meet to the end of a string
			if ((str_a[i] == 0) || (str_b[i] == 0)) done = 1;
			else if (str_a[i] != str_b[i])
			{
				match = i + 1;
				if (((int)str_a[i] - (int)str_b[i]) < 0)
					match = 0 - (i + 1);
			}
			
			i++;
		}
		
		return match;
	}

	template <typename Tuple>
	__host__ __device__
	void operator() (const Tuple t)
	{
		long data = thrust::get<0>(t);
		double lat = thrust::get<1>(t);

		if ( data >= start && data <= end)
			thrust::get<2>(t) = lat;							
		else
			thrust::get<2>(t) = -1;
	}

};

struct space_between
{
	double left_lon, left_lat, right_lon, right_lat;
	
	__host__ __device__
	space_between(double _lf_lon, double _lf_lat, double _rt_lon, double _rt_lat):
		left_lon(_lf_lon), left_lat(_lf_lat), right_lon(_rt_lon), right_lat(_rt_lat) {}
	
	template <typename Tuple>
	__host__ __device__	
	void operator() (const Tuple t)
	{
		//printf("----------------------\n");
		double lon = thrust::get<0>(t);
		double lat = thrust::get<1>(t);
		long id = thrust::get<2>(t);
		//printf("=============%ld\n", id);
		if (lon < right_lon && lon > left_lon && lat > right_lat && lat < left_lat)
			thrust::get<3>(t) = id;
		else
			thrust::get<3>(t) = -1;
	}
};

struct space_time_between
{
	double left_lon, left_lat, right_lon, right_lat;
	long start, end;

	__host__ __device__
	space_time_between(double _lf_lon, double _lf_lat, double _rt_lon, double _rt_lat, long _start, long _end):
		left_lon(_lf_lon), left_lat(_lf_lat), right_lon(_rt_lon), right_lat(_rt_lat), start(_start), end(_end) {}

	template <typename Tuple>
	__host__ __device__
	void operator() (const Tuple t)
	{
		double lon = thrust::get<0>(t);
		double lat = thrust::get<1>(t);
		long time = thrust::get<2>(t);
		long id = thrust::get<3>(t);
		

		if (lon < right_lon && lon > left_lon && lat > right_lat && lat < left_lat && time > start && time < end)
			thrust::get<4>(t) = id;
		else 
			thrust::get<4>(t) = -1;
	}
};

struct points_to_traj
{
	int		tid;
	int		*tid_data;
	double 		*lon_data, *lon_res;
	double 		*lat_data, *lat_res;
	long   		*ts_data, *ts_res;

	points_to_traj(int _tid, int *_tids, double *_lon, double *_lat, long *_ts, double *_res_lon, double *_res_lat, long *_res_ts) : tid(_tid), tid_data(_tids), lon_data(_lon), 
				lat_data(_lat), ts_data(_ts), lon_res(_res_lon), lat_res(_res_lat), ts_res(_res_ts) {}

	__host__ __device__
	void operator()(const int& i) 
	{
		//if (tid_data[i] != -1)	printf("--------------%d\n line 217", tid_data[i]);
		if (tid_data[i] == tid)
		{	
			lon_res[i] = lon_data[i];	
			//printf("----------pt-- to traj %ld\n", ts_data[i]);
			lat_res[i] = lat_data[i];
			ts_res[i] = ts_data[i];
		}
	}
};
#endif
