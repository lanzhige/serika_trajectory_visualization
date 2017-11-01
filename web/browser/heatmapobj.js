/**
 * Created by IDMV61 on 11/3/2016.
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
            require("heatmap.js")
        );
    } else {
        // Browser globals (root is window)
        context[name] = factory(0);
    }
})("HeatMapObj",this, function() {
    'use strict';

    var HeatMapObj=function(heatmap,cfg){
        this.heatmap=heatmap;
        this.initialize(cfg);
    };

    HeatMapObj.prototype.initialize= function(cfg){
        this.cfg=Object.assign({},cfg);
        this.dataset=[];
        this.on_display_index=undefined;
        //this.grids={lat:[],lng:[]};
    };

    HeatMapObj.prototype.addData=function(data){
        this.dataset.push(Object.assign([],data));
    };

    HeatMapObj.prototype.setData=function(dataset){
        this.dataset=Object.assign({},dataset);
    };

    HeatMapObj.prototype.removeData=function(index){
        this.dataset[index].pop();
    };

    HeatMapObj.prototype.update=function(cfg,data){
        Object.assign(this.cfg,cfg);
        if (data!=undefined)
            this.dataset=Object.assign({},data);
    };

    HeatMapObj.prototype.show=function(index){
        this.on_display_index=index;
        this.display();
    };

    HeatMapObj.prototype.hide=function(){
        console.log(this.heatmap.container);
        $(this.heatmap.container).hide();
    };

    HeatMapObj.prototype.display=function(){
        this.heatmap.setData({
            min:0,
            max:10,
            data:this.dataset[this.on_display_index]
        })
    };

    HeatMapObj.prototype.removeAll=function(){
        this.dataset=[];
    };

    return HeatMapObj;
});