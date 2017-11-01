#include "wrapper.h"
#include "cuda/heatmap.h"
#include "cuda/radar.h"
#include "cuda/edgebundling.h"
#include "cuda/selector.h"
#include "cuda/geoArray.h"

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

#include <vector>
#include <iostream>
#include <string>
#include <dirent.h>
#include <string.h>
#include <stdio.h>

#include <thrust/count.h>

typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::lib::bind;
using selika::heatmap::HeatMap;
using selika::radar::Radar;
using std::vector;
using selika::GeoArray;
//using selika::ChordMatrix;
using selika::Selector;

comdb db;
GeoArray geo_array_;
// pull out the type of messages sent by our config
typedef server::message_ptr message_ptr;

// Define a callback to handle incoming messages
void on_message(server* s, websocketpp::connection_hdl hdl, message_ptr msg) {
  selika::MessageWrapper messageWrapper(msg->get_payload());
  switch(messageWrapper.op){
    case 0:{
      selika::HeatMapWrapper heatMapWrapper(msg->get_payload());

      HeatMap hp(heatMapWrapper.lat_top, heatMapWrapper.lat_bottom,
          heatMapWrapper.lng_left, heatMapWrapper.lng_right,
          heatMapWrapper.grids_row, heatMapWrapper.grids_col,
          heatMapWrapper.start_time,heatMapWrapper.end_time,
          heatMapWrapper.time_window,heatMapWrapper.time_interval);

      int length =(ceil((heatMapWrapper.end_time-heatMapWrapper.start_time
          -heatMapWrapper.time_window)/heatMapWrapper.time_interval)+1)
          *heatMapWrapper.grids_row*heatMapWrapper.grids_col;

      selika::HeatMapDataWrapper heatMapDataWrapper(
          hp.query_heatmap(db),length);
      s->send(hdl, heatMapDataWrapper.data_str, msg->get_opcode());
      break;
    }
    case 1:{
      selika::RadarWrapper radarWrapper(msg->get_payload());

      Radar radar(radarWrapper.start_time,radarWrapper.end_time
          , radarWrapper.time_window, radarWrapper.time_interval
          , radarWrapper.r,radarWrapper.lat,radarWrapper.lng
          , radarWrapper.arg);
      radar.init();
      int length=(1+ceil(radarWrapper.end_time-radarWrapper.start_time
          -radarWrapper.time_window)/radarWrapper.time_interval)
          *radarWrapper.arg;
      selika::RadarDataWrapper radarDataWrapper(
          radar.query_radar(db),length,radarWrapper.id);
      s->send(hdl, radarDataWrapper.data_str,msg->get_opcode());
      break;
    }
    case 2:{
      selika::EdgeBundlingWrapper edgeBundlingWrapper(msg->get_payload());

      vector<selika::edgebundling::Edge> data_edges;
      selika::edgebundling::dataFetch(db,&data_edges
          , edgeBundlingWrapper.lat_top, edgeBundlingWrapper.lat_bottom
          , edgeBundlingWrapper.lng_left, edgeBundlingWrapper.lng_right
          , edgeBundlingWrapper.start_time, edgeBundlingWrapper.end_time
          , edgeBundlingWrapper.time_window
          , edgeBundlingWrapper.time_interval);

      selika::edgebundling::EdgeBundling edge_bundling(
          data_edges, edgeBundlingWrapper.K
          , edgeBundlingWrapper.S, edgeBundlingWrapper.P
          , edgeBundlingWrapper.P_rate, edgeBundlingWrapper.C
          , edgeBundlingWrapper.I, edgeBundlingWrapper.I_rate
          , edgeBundlingWrapper.threshold);
      edge_bundling.forceBundle();
      selika::EdgeDataWrapper edge_data_wrapper(edge_bundling.outputPoints());
      s->send(hdl, edge_data_wrapper.data_str, msg->get_opcode());
      break;
    }
    case 3:{
      selika::GeoObjWrapper geo_obj_wrapper_(msg->get_payload());
      if (geo_obj_wrapper_.radius>0){
        geo_array_.add(geo_obj_wrapper_.id, geo_obj_wrapper_.cnt_lat
            , geo_obj_wrapper_.cnt_lng, geo_obj_wrapper_.radius
            , geo_obj_wrapper_.start_time, geo_obj_wrapper_.end_time);
      } else/* if (geo_obj_wrapper_.left_lng>0){
          GeoObj geo_obj(geo_obj_wrapper_.id, geo_obj_wrapper_.left_lng
              , geo_obj_wrapper_.right_lng, geo_obj_wrapper_.top_lat
              , geo_obj_wrapper_.bottom_lat);
          geo_obj_list.push_back(geo_obj);
        } else*/ {
          std::cerr<<"undefined geo-object"<<std::endl;
          break;
        }
        //add geo_obj.selectData
        break;
      }
      case 4:{
        selika::ChordMatrixWrapper chord_matrix_wrapper_(msg->get_payload());
        /*selika::ChordGenerator c_generator_(chord_matrix_wrapper_.id
            , raw_pointer_cast(db.col_id_vec.data())
            , raw_pointer_cast(db.col_time_vec.data())
            , raw_pointer_cast(db.col_lat_vec.data())
            , raw_pointer_cast(db.col_lon_vec.data())
            , db.col_id_vec.size(), &geo_obj_list[0]
            , geo_obj_list.size()
        );*/
printf("center id: %d\n",chord_matrix_wrapper_.id);
        geo_array_.selectData(db,chord_matrix_wrapper_.id);
        //selika::calculateChordMatrix(chord_matrix_);
        //std::cerr<<"data_len: "<<*(chord_matrix_.out_len)<<std::endl;
        //std::cerr<<"data_len: "<<*(chord_matrix_.out_len)<<std::endl;
        break;
      }
      case 5:{
        selika::ChordWrapper chord_wrapper_(msg->get_payload());
        geo_array_.genChord(chord_wrapper_.id);
        selika::ChordDataWrapper chord_data_wrapper_(
            chord_wrapper_.id, geo_array_.chord, geo_array_.chord_len
        );
printf("chord_len: %d\n",geo_array_.chord_len);
        s->send(hdl, chord_data_wrapper_.data_str, msg->get_opcode());
        break;
      }
      case 6:{
        geo_array_.clear();
        break;
      }
      case 7:{
        selika::ChordMatrixWrapper chord_matrix_wrapper_(msg->get_payload());
        geo_array_.remove(chord_matrix_wrapper_.id);
        break;
      }
      case 11:{
        selika::POIWrapper POI_wrapper_(msg->get_payload());
        break;
      }
      default:
        std::cerr<<"unexpected operation!"<<std::endl;
    }
      //s->send(hdl, heatMapDataWrapper.data_str, msg->get_opcode());  

    // check for a special command to instruct the server to stop listening so
    // it can be cleanly exited.
    if (msg->get_payload() == "stop-listening") {
        s->stop_listening();
        return;
    }
}

int sortDB(comdb *db){
  selika::GlobalData global_data_(db);
  global_data_.sortByIdTime();
  return 0;
}

int main() {
    // Create a server endpoint
    server echo_server;
    cudaSetDevice(1);
    DIR *dir;
    struct dirent *ent;
    char path[50];
    if (( dir = opendir("./data") ) != NULL)
    {
      while (( ent = readdir(dir) ) != NULL)
      {
        // skip self and parent
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
          continue;

        if (strlen("./data") + strlen(ent->d_name) + 2 > sizeof(path))
          fprintf(stderr, "name %s %s too long\n", "./data", ent->d_name);
        else
        {
          sprintf(path, "%s/%s", "./data", ent->d_name);
          // std::cout << path << "\n";
          load_data(path, db);


          //cudaDeviceSynchronize();	
        }
      }
      closedir(dir);
    }
    else
    {
      perror("read file error!");
      return EXIT_FAILURE;
    }
    std::cerr<<"file read finished!"<<std::endl;
sortDB(&db);
/*int len = db.col_id_vec.size();
int *id = new int[len];
long *time = new long[len];

thrust::copy_n(db.col_id_vec.begin(), len, id);
thrust::copy_n(db.col_time_vec.begin(), len, time);
for (int i = 0;i<len;i++) printf("id :%d, time: %ld\n",id[i], time[i]);
*/
    try {
        // Set logging settings
        echo_server.set_access_channels(websocketpp::log::alevel::all);
        echo_server.clear_access_channels(websocketpp::log::alevel::frame_payload);

        // Initialize Asio
        echo_server.init_asio();

        // Register our message handler
        echo_server.set_message_handler(bind(&on_message, &echo_server,
                websocketpp::lib::placeholders::_1, websocketpp::lib::placeholders::_2));

        // Listen on port 8080
        echo_server.listen(9002);

        // Start the server accept loop
        echo_server.start_accept();

        // Start the ASIO io_service run loop
        echo_server.run();
    } catch (websocketpp::exception const & e) {
        std::cout << e.what() << std::endl;
    } catch (...) {
        std::cout << "other exception" << std::endl;
    }
}
