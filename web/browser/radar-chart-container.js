/**
 * Created by IDMV61 on 11/17/2016.
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
            require("radar-chart.js"),
            require("radar-chart-cfg.js"),
            require("line-chart.js"),
            require("jquery.js"),
            require("moment.js")
        );
    } else {
        // Browser globals (root is window)
        context[name] = factory(0);
    }
})("RadarChartContainer",this, function() {
    'use strict';

    var RadarChartContainer=function(cfg){
        this.initialize(cfg.msg,cfg.cfg,cfg.title);
    };

    RadarChartContainer.prototype.initialize= function(msg,cfg){
        var default_msg=jQuery.extend(true,{},DEFAULT_MSG);
        var default_cfg=jQuery.extend(true,{},DEFAULT_CFG);
        this.msg=jQuery.extend(true,default_msg,msg);
        this.cfg=jQuery.extend(true,default_cfg,cfg);
        this.id=cfg.id;
        this.title=this.id;
        this.dom=undefined;
        this.canvas=undefined;
        this.visible=undefined;
        this.radar_set=[];
        this.slider=undefined;
        this.chart=undefined;
        this.radar_cfg=undefined;
        this.line_chart=undefined;
        this.max_value=0;
    };

    RadarChartContainer.prototype.addRadar=function(radar){
        for (var i=0;i<radar.cfg.data.datasets[0].data.length;i++)
            if (this.max_value<radar.cfg.data.datasets[0].data[i]) this.max_value=radar.cfg.data.datasets[0].data[i];
        this.radar_set.push(radar);
        radar.bind(this);
    };

    RadarChartContainer.prototype.update=function(cfg){
        this.cfg=jQuery.extend(true,this.cfg,cfg);
    };

    RadarChartContainer.prototype.remove=function(){
        if (this.dom!=undefined) {
            if (this.line_chart!=undefined) this.line_chart.remove();
            this.dom.remove();
        }else return console.log("error in removing radar-container");
    };

    RadarChartContainer.prototype.hide=function(){
        this.dom.hide();
    };

    RadarChartContainer.prototype.show=function(){
        if (this.dom==undefined){
            var div=$("#wrapper").append("<div id='radar-div"+this.title+"' class='col-md-4' hidden>"+
            "<div class='ibox float-e-margins'>"+
                "<div class='ibox-title'>"+
                "<h5>Radar Chart</h5>"+
                "<button id='radar-button"+this.title+"' style='position: absolute; top:10px; right:30px'>Line Chart</button>"+
            "</div>"+
            "<div class='ibox-content'>"+
                "<canvas id='radar-canvas"+this.title+"' width='400' height='400'></canvas>"+
                "</div>"+
                "</div>"+
                "</div>");
            this.dom=$("#radar-div"+this.title);
            this.canvas=$("#radar-canvas"+this.title);
            this.dom.css(this.cfg);
            this.dom.draggable();
            this.addToggleLineChart();
        }
        this.dom.show();
    };

    RadarChartContainer.prototype.setVisible=function(index){
        if (this.radar_set[index]==undefined) {
            console.log("current visible radar chart undefined.");
            return;
        }
        this.visible=index;
    };

    RadarChartContainer.prototype.display=function(){
        if (this.radar_set.length==0) {
            console.log("No Data Added To Be Displayed as Radar Chart!");
            return;
        }
        if (this.visible==undefined) this.setVisible(0);
        this.show();
        this.radar_set[this.visible].show();
    };

    const DEFAULT_CFG={
        "left": "230px",
        "z-index":"104",
        "width":"465px",
        "position":"absolute",
        "top":"140px",
        "opacity":0.9
    };

    const DEFAULT_MSG={
        "lat":undefined,
        "lng":undefined,
        "r":undefined,
        "start_time":undefined,
        "end_time":undefined,
        "time_window":undefined,
        "time_interval":undefined,
        "arg":undefined
    };

    RadarChartContainer.prototype.randomColorFactor=function(){
        return Math.round(Math.random()*255);
    };

    RadarChartContainer.prototype.randomColor=function(opacity){
        return 'rgba(' + this.randomColorFactor() + ',' + this.randomColorFactor()
          + ',' + this.randomColorFactor() + ',' + (opacity || '.3') + ')';
    };

    RadarChartContainer.prototype.setRandomColor=function(){
        return Object.assign({},{
            borderColor:this.randomColor(0.5),
            backgroundColor:this.randomColor(0.5),
            pointBorderColor:this.randomColor(0.7),
            pointBackgroundColor:this.randomColor(0.5)
        });
    };

    RadarChartContainer.prototype.addToggleLineChart=function(){
        if (this.radar_set.length>0) var labels=JSON.parse(JSON.stringify(this.radar_set[0].cfg.data.labels));
        var dataset=[];

        for (var i=0;i<labels.length;i++){
            dataset.push({
                type:"spline",
                name:labels[i],
                data:[]
            });
            var index=dataset.length-1;
            for (var j=0;j<this.radar_set.length;j++){
                dataset[index].data.push(this.radar_set[j].cfg.data.datasets[0].data[i]);
            }
        }
        var bar={
            type:"column",
            name:"average",
            data:[]
        };
        for (i=0;i<this.radar_set.length;i++){
            var average=0;
            for (j=0;j<labels.length;j++)
                average+=this.radar_set[i].cfg.data.datasets[0].data[j];
            bar.data.push(Math.round(average/8));
        }
        var that=this;
        $("#radar-button"+that.title).click(function(){
            if (that.line_chart==undefined) {
                that.line_chart = new LineChart(that.id, {});

                for (var i = 0; i < labels.length; i++) {
                    that.line_chart.addData(dataset[i]);
                }
                that.line_chart.setPlotOptions({
                    start_time: that.msg.start_time,
                    end_time: that.msg.end_time,
                    time_window: that.msg.time_window,
                    time_interval: that.msg.time_interval
                });
            }
            that.line_chart.show();
        });
    };

    RadarChartContainer.prototype.setConfig=function(cfg){
        this.radar_cfg=jQuery.extend(true,this.radar_cfg,{
            options:{
                scale:{
                    ticks:{
                        fixedStepSize:Math.ceil(this.max_value/7),
                        max:Math.ceil(this.max_value/7)*7
                    }
                }
            }
        },cfg);
    };

    var DEFAULT_DATA={
        label:"",
        backgroundColor: "rgba(179,181,198,0.2)",
        borderColor: "rgba(179,181,198,1)",
        pointBackgroundColor: "rgba(179,181,198,1)",
        pointBorderColor: "#fff",
        pointHoverBackgroundColor: "#fff",
        pointHoverBorderColor: "rgba(179,181,198,1)",
        data:[]
    };

    const RGD_SET=[
        "31,120,180",
        "166,206,227",
        "51,160,44",
        "178,223,138",
        "227,26,28",
        "251,154,153",
        "255,127,0",
        "253,191,111"
    ];

    return RadarChartContainer;
});