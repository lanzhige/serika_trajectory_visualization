/**
 * Created by idmv61 on 10/6/2016.
 */
;(function(name,context, factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define([], factory);
    } else if (typeof module === 'object' && module.exports) {
        // Node. Does not work with strict CommonJS, but
        // only CommonJS-like environments that support module.exports,
        // like Node.
        module.exports = factory(
            require("chart.js")
        );
    } else {
        // Browser globals (root is window)
        context[name] = factory(0);
    }
})("RadarChart",this, function() {
    'use strict';

    var RadarChart=function(cfg){
        this.initialize(cfg||{});
    };

    RadarChart.prototype.initialize= function(cfg){
        this.cfg=Object.assign({},{
            type:'radar',
            data:{
                labels:['N','NE','E','SE','S','SW','W','NW']
            },
            options:{
                title:{
                    display:true,
                    text:"radar chart"
                },
                scale:{
                    reverse:false
                }
            }
        },cfg);
        Object.assign(this.cfg.data,{datasets:[]});
        this.msg=[];
        this.dataset=[];
        this.container=undefined;
    };

    RadarChart.prototype.addData=function(data){
        var temp_data=Object.assign({},DATASET_TEMPLATE);
        Object.assign(temp_data,{data:data});
        this.cfg.data.datasets.push(temp_data);
    };

    RadarChart.prototype.setData=function(msg,data){
        this.msg=Object.assign({},msg);
        this.cfg.data.datasets=Object.assign({},data);
    };

    RadarChart.prototype.removeData=function(){
        this.msg=[];
        this.cfg=Object.assign({},RADAR_CFG_TEMPLATE);
    };

    RadarChart.prototype.update=function(cfg){
        Object.assign(this.cfg,cfg);
        if (this.container.visible_chart!=undefined) this.chart.update();
    };

    RadarChart.prototype.bind=function(container){
        this.container=container;
    };

    RadarChart.prototype.show=function(){
        this.container.setConfig(this.cfg);
        if (this.container.chart==undefined) {
            this.container.chart=new Chart(this.container.canvas,this.container.radar_cfg);
        }else{
            //this.container.chart.config=this.container.radar_cfg;
            this.container.chart.update();
        }
    };

    var DATASET_TEMPLATE={
        label:"radar chart",
        backgroundColor: "rgba(179,181,198,0.2)",
        borderColor: "rgba(179,181,198,1)",
        pointBackgroundColor: "rgba(179,181,198,1)",
        pointBorderColor: "#fff",
        pointHoverBackgroundColor: "#fff",
        pointHoverBorderColor: "rgba(179,181,198,1)"
    };

    var RADAR_CFG_TEMPLATE = {
        type:'radar',
        data:{
            labels:['0','45','90','135','180','235','270','315']
        },
        options:{
            title:{
                display:false
            },
            labels:{
                display:false
            },
            scale:{
                reverse:false,
                ticks:{
                    display:false,
                    ticks:{
                        beginAtZero:true
                    }
                }
            },
            hover:{
                mode:'single'
            }
        }
    };

    return RadarChart;
});