/**
 * Created by IDMV61 on 11/21/2016.
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
            require("radar-chart.js")
        );
    } else {
        // Browser globals (root is window)
        context[name] = factory(0);
    }
})("RadarChartCfg",this, function() {
    'use strict';

    var RadarChartCfg=function(cfg){
        this.initialize(cfg.msg,cfg.cfg,cfg.title);
    };

    RadarChartCfg.prototype.initialize= function(msg,cfg){
        this.msg=Object.assign(DEFAULT_MSG,msg);
        this.cfg=Object.assign(DEFAULT_CFG,cfg);
        this.title="";
        this.dom=undefined;
        this.visible=false;
        this.radar_set=[];
        this.slider=undefined;
    };

    RadarChartCfg.prototype.addRadar=function(radar){
        this.radar_set.push(radar);
        radar.bind(this);
    };

    RadarChartCfg.prototype.update=function(cfg){
        this.cfg=Object.assign(this.cfg,cfg);
    };

    RadarChartCfg.prototype.remove=function(){
        if (this.dom!=undefined) return this.dom.remove(); else return -1;
    };

    RadarChartCfg.prototype.hide=function(){
        this.dom.hide();
    };

    RadarChartCfg.prototype.show=function(){
        if (this.dom==undefined){
            var div=$(window).append("<div class='col-md-4' hidden>"+
                "<div class='ibox float-e-margins'>"+
                "<div class='ibox-title'>"+
                "<h5>Radar Chart</h5>"+
                "</div>"+
                "<div class='ibox-content'>"+
                "<canvas id='radar_canvas'"+this.title+" width='400' height='400'></canvas>"+
                "</div>"+
                "</div>"+
                "</div>");
        }else this.dom.show();
    };

    var DEFAULT_CFG={
        type:'radar',
        data:{
            labels:[],
            datasets:[]
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
    };

    return RadarChartCfg;
});