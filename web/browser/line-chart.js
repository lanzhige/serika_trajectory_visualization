/**
 * Created by IDMV61 on 12/6/2016.
 */
;(function(name,context, factory) {
    if (typeof define === 'function' && define.amd) {
        // AMD. Register as an anonymous module.
        define([], factory);
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory(
            require("bootstrap.js"),
            require("jquery.js"),
            require("highcharts.js")
        );
    } else {
        context[name] = factory(0);
    }
})("LineChart",this, function() {
    'use strict';

    var LineChart=function(id,cfg){
        this.initialize(id,cfg);
    };

    LineChart.prototype.initialize=function(id,cfg){
        this.title=id;
        var default_cfg=jQuery.extend(true,{},LINE_CHART_CFG);
        this.cfg=jQuery.extend(true,default_cfg,cfg);
        this.dom=undefined;
        this.chart=undefined;
    };

    LineChart.prototype.addData=function(data){
        if (this.cfg!=undefined){
            this.cfg.series.push(jQuery.extend(true,{name:"",data:[]},data));
        }else console.log("Failed to add data to line chart, line chart config uninitialized!");
    };

    LineChart.prototype.show=function(){
        if (this.dom==undefined){
            var div=$("#wrapper").append("<div id='line-div"+this.title+"' class='col-md-4' hidden>"+
                "<div class='ibox float-e-margins'>"+
                "<div class='ibox-title'>"+
                "<h5>Line Chart</h5>"+
                "<button id='line-button-close"+this.title+"' type='button' class='fa fa-close' style='position: absolute; right:30px; top:10px'></button>"+
                "<button id='line-button-drag"+this.title+"' type='button' class='fa fa-arrows' style='position: absolute; right:60px; top:10px'></button>"+
                "</div>"+
                "<div class='ibox-content'>"+
                "<div id='line-canvas"+this.title+"' style='width:1200px; height:600px'></div>"+
                "</div>"+
                "</div>"+
                "</div>");
            this.dom=$("#line-div"+this.title);
            this.dom.css(DEFAULT_CSS);
        }
        this.dom.show();
        var that = this;
        that.dom.draggable({
            cancel: "div.ibox-content",
            drag: function (e, ui) {
                $("#line-button-drag"+that.title).trigger('drag');
            }
        });
        $("#line-button-close"+this.title).click(function(){that.dom.hide()});
        this.display();
    };

    LineChart.prototype.display=function(){
        if (this.chart==undefined) {
            this.cfg.chart.renderTo='line-canvas'+this.title;
            console.log(this.cfg);
            this.chart = new Highcharts.Chart(this.cfg);
        }else {
            this.chart.update();
        }
    };

    LineChart.prototype.remove=function(){
        if (this.dom!=undefined)
            this.dom.remove();
        else
            console.log("chart-dom is not defined!");
    };

    LineChart.prototype.setPlotOptions=function(cfg){
        this.cfg.plotOptions.spline.pointInterval=cfg.time_interval*1000;
        this.cfg.plotOptions.spline.pointStart=cfg.start_time*1000;
    };

    LineChart.prototype.setLabels=function(cfg){
        var chart_num=Math.ceil((cfg.end_time-cfg.start_time)/(24*3600));
        var canvas=$("#line-canvas"+this.title);
        line_chart_css["max-width"]=""+canvas.width()+"px";
        line_chart_css["height"]=""+canvas.height()+"px";

        for (var i=0;i<chart_num;i++){

        }

        setPlotOptions(cfg);
    };

    const DEFAULT_CSS={
        "left": "130px",
        "z-index":"104",
        "width":"1265px",
        "position":"absolute",
        "top":"70px",
        "opacity":0.9
    };

    var line_chart_css={
        "min-width": "800px",
        "max-width": "1200px",
        "height": "600px",
        "margin": "0 auto"
    };

    const LINE_CHART_CFG= {
        chart: {
            zoomType:'x'
        },
        title: {
            text: 'Line Chart'
        },
        xAxis: {
            type: 'datetime',
            labels: {
                overflow: 'justify'    //deprecated
            }
        },
        yAxis: {
            title: {
                text: 'Taxi Number'
            },
            minorGridLineWidth: 1,
            gridLineWidth: 1,
            alternateGridColor: null,
            plotLines: [{
                value:0,
                width:1,
                color:'#808080'
            }]
        },
        tooltip: {
            valueSuffix: ' Cars'
        },
        plotOptions: {
            spline: {
                pointInterval: 3600000, // one hour
                pointStart: Date.UTC(2015, 4, 31, 0, 0, 0)
            }
        },
        legend: {
            layout: 'vertical',
            align: 'right',
            verticalAlign: 'middle',
            borderWidth: 0
        },
        series: [],
        navigation: {
            menuItemStyle: {
                fontSize: '10px'
            }
        }
    };

    return LineChart;
});