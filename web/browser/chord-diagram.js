/**
 * Created by IDMV61 on 2/21/2017.
 */
var ChordDiagram = function(matrix,index_set,container,cnt_pixel){
    this.matrix = jQuery.extend(true,[],matrix);
    //the data matrix of in and out. e.g.: matrix[i][j] means the value come in from i and go out to j
    this.index_array = jQuery.extend(true,[],index_set);
    //the array of geographical objects. contains the latitude and longitude of center of each object.
    this.cnt_x = cnt_pixel.x;
    this.cnt_y = cnt_pixel.y;
    //x, y are the pixel coordinate
    this.container = container;
    this.genMatrix(0);
    // 0 is the threshold to display, if the value is below threshold the ribbon will not be generated
    this.genChord();
};

ChordDiagram.prototype.compareValue = function(compare) {
    return function(a, b) {
        return compare(
            a.source.value + a.target.value,
            b.source.value + b.target.value
        );
    };
};

ChordDiagram.prototype.range = function(start, stop, step) {
    start = +start; stop = +stop;
    step = (n = arguments.length) < 2 ? (stop = start, start = 0, 1) : n < 3 ? 1 : +step;

    var i = -1,
        n = Math.max(0, Math.ceil((stop - start) / step)) | 0,
        range = new Array(n);

    while (++i < n) {
        range[i] = start + i * step;
    }

    return range;
};

ChordDiagram.prototype.chord = function(){
    var padAngle = 0,
        sortGroups = null,
        sortSubgroups = null,
        sortChords = null;

    function chord(matrix,angle_array) {
        //function to calculate the sorted array of angle and the sum of chord value to decide the size of ribbon.

        var n = matrix.length,
            groupSums = [],
            groupIndex = range(n),
            subgroupIndex = [],
            chords = [],
            groups = chords.groups = new Array(n),
            subgroups = new Array(n * n *2),
            k,
            x,
            x0,
            dx,
            i,
            j;

        // Compute the sum.
        k = 0; i = -1;
        while (++i < n){
            x=0; j=-1;
            while (++j < n){
                x+= matrix[i][j];
                if (i!=j) x+=matrix[j][i];
            }
            groupSums.push(x);
            subgroupIndex.push(range(n));
            k += x;
        }

        // Sort groups…
        if (sortGroups) groupIndex.sort(function(a, b) {
            return sortGroups(groupSums[a], groupSums[b]);
        });

        // Sort subgroups…
        if (sortSubgroups) subgroupIndex.forEach(function(d, i) {
            d.sort(function(a, b) {
                return sortSubgroups(matrix[i][a], matrix[i][b]);
            });
        });

        // Convert the sum to scaling factor for [0, 2pi].
        k = Math.max(0, Math.PI - padAngle * n) / k;
        dx = k ? padAngle : Math.PI / n;

        // Compute the start and end angle for each group and subgroup.
        // Note: Opera has a bug reordering object literal properties!
        i = -1;

        while (++i < n) {
            x0 = angle_array[i];
            j = i;
            x = x0;
            while (--j >=0) {
                var di = groupIndex[i],
                    dj = subgroupIndex[di][j],
                    v = matrix[di][dj],
                    a0 = x,
                    a1 = x+= v * k;
                subgroups[dj * n + di] = {//---calculate in and out
                    index: di,
                    subindex: dj,
                    startAngle: a0,
                    endAngle: a1,
                    value: v
                };
                if (dj!=di){
                    v = matrix[dj][di];
                    a0 = x;
                    a1= x+= v * k;
                    subgroups[n * n + di*n +dj]={
                        index: di,
                        subindex: dj,
                        startAngle:a0,
                        endAngle:a1,
                        value:v
                    }
                }
            }
            j=i;
            di = groupIndex[i];
            dj = subgroupIndex[di][j];
            v = matrix[di][dj];
            a0 = x;
            a1 = x+= v * k;
            subgroups[dj * n + di] = {//---need to calculate in and out
                index: di,
                subindex: dj,
                startAngle: a0,
                endAngle: a1,
                value: v
            };
            j=n;
            while (--j>i){
                di = groupIndex[i];
                dj = subgroupIndex[di][j];
                v = matrix[di][dj];
                a0 = x;
                a1 = x+= v * k;
                subgroups[dj * n + di] = {//---need to calculate in and out
                    index: di,
                    subindex: dj,
                    startAngle: a0,
                    endAngle: a1,
                    value: v
                };
                if (dj!=di){
                    v = matrix[dj][di];
                    a0 = x;
                    a1= x+= v * k;
                    subgroups[n * n + di*n +dj]={
                        index: di,
                        subindex: dj,
                        startAngle:a0,
                        endAngle:a1,
                        value:v
                    }
                }
            }
            groups[di] = {
                index: di,
                startAngle: x0,
                endAngle: x,
                value: groupSums[di]
            };
            //x += dx;
        }

        // Generate chords for each (non-empty) subgroup-subgroup link.
        i = -1; while (++i < n) {
            j=i;
            while (j>=0){
                var source = subgroups[j * n + i],
                    target = (i==j)?subgroups[j * n + i]:subgroups[n*n + j*n +i];
                if (source.value) {
                    chords.push({source: source, target: target});
                }
                j--;
            }
            j=n;
            while (--j>i){
                source = subgroups[j * n + i];
                target = (i==j)?subgroups[j * n + i]:subgroups[n*n + j*n +i];
                if (source.value) {
                    chords.push({source: source, target: target});
                }
            }
        }
        return sortChords ? chords.sort(sortChords) : chords;
    }

    chord.compareValue = function(compare) {
        return function(a, b) {
            return compare(
                a.source.value + a.target.value,
                b.source.value + b.target.value
            );
        };
    };

    var range = function(start, stop, step) {
        start = +start; stop = +stop;
        step = (n = arguments.length) < 2 ? (stop = start, start = 0, 1) : n < 3 ? 1 : +step;

        var i = -1,
            n = Math.max(0, Math.ceil((stop - start) / step)) | 0,
            range = new Array(n);

        while (++i < n) {
            range[i] = start + i * step;
        }

        return range;
    };

    chord.padAngle = function(_) {
        return arguments.length ? (padAngle = Math.max(0, _), chord) : padAngle;
    };

    chord.sortGroups = function(_) {
        return arguments.length ? (sortGroups = _, chord) : sortGroups;
    };

    chord.sortSubgroups = function(_) {
        return arguments.length ? (sortSubgroups = _, chord) : sortSubgroups;
    };

    chord.sortChords = function(_) {
        return arguments.length ?
            (_ == null ? sortChords = null : (sortChords = this.compareValue(_))._ = _, chord) : sortChords && sortChords._;
    };

    return chord;
};

ChordDiagram.prototype.genChord = function() {
    var svg = d3.select(this.container).append("svg").attr("class","chord-svg");
    var width = +parseInt(svg.style("width"));
    var height = +parseInt(svg.style("height"));
    width = 250;
    height = 250;
    var outerRadius = Math.min(width, height) * 0.5 - 40;

    var innerRadius = outerRadius - 10;

    var chordDiagram = this.chord()
        .padAngle(0.05);
        //.sortSubgroups(d3.descending)
        //.sortChords(d3.descending);

    var arc = d3.arc()
        .innerRadius(innerRadius)
        .outerRadius(outerRadius);

    var ribbon = d3.ribbon()
        .radius(innerRadius);

    var color = d3.scaleOrdinal()
        .domain(d3.range(11))
        .range(["#ffff99","#1f78b4","#b2df8a","#33a02c","#fb9a99"
            ,"#e31a1c","#fdbf6f","#ff7f00","#cab2d6","#6a3d9a","#a6cee3"]);
    //color of each chord and ribbon

    var chordData = chordDiagram(this.matrix,this.index_array);

    console.log(chordData);
    console.log(svg);

    var g = svg.append("g")
        .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")")
        .datum(chordData);

    var group = g.append("g")
        .attr("class", "groups")
        .selectAll("g")
        .data(function (chords) {
            return chords.groups;
        })
        .enter().append("g");

    group.append("path")
        .style("fill", function (d) {
            return color(d.index);
        })
        .style("stroke", function (d) {
            return d3.rgb(color(d.index)).darker();
        })
        .attr("d", arc)
        .on("mouseover", fade(0.2))
        .on("mouseout", fade(1));


    //var formatValue = d3.formatPrefix(",.0", 1e2);
    var formatValue = d3.formatPrefix(",.0", 10);
    var groupTick = group.selectAll(".group-tick")
        .data(function (d) {
            return groupTicks(d, 10);
        })
        .enter().append("g")
        .attr("class", "group-tick")
        .attr("transform", function (d) {
            return "rotate(" + (d.angle * 180 / Math.PI - 90) + ") translate(" + outerRadius + ",0)";
        });

    groupTick.append("line")
        .attr("x2", 6);
    groupTick
        .filter(function (d) {
            return d.value % 10 === 0;
        })
        .append("text")
        .attr("x", 8)
        .attr("dy", ".35em")
        .attr("transform", function (d) {
            return d.angle > Math.PI ? "rotate(180) translate(-16)" : null;
        })
        .style("text-anchor", function (d) {
            return d.angle > Math.PI ? "end" : null;
        })
        .text(function (d) {
            return formatValue(d.value);
        });

    g.append("g")
        .attr("class", "ribbons")
        .selectAll("path")
        .data(function (chords) {
            return chords;
        })
        .enter().append("path")
        .attr("d", ribbon)
        .style("fill", function (d) {
            return color(d.target.index);
        })
        .style("stroke", function (d) {
            return d3.rgb(color(d.target.index)).darker();
        });

// Returns an array of tick angles and values for a given group and step.
    function groupTicks(d, step) {
        var k = (d.endAngle - d.startAngle) / d.value;
        return d3.range(0, d.value, step).map(function (value) {
            return {value: value, angle: value * k + d.startAngle};
        });
    }

    function fade(opacity) {
        return function (g) {
            svg.selectAll("g.ribbons path")
                .filter(function (d) {
                    console.log(d);
                    return d.source.index != g.index && d.target.index != g.index;
                })
                .transition()
                .style("opacity", opacity);
        };
    }
};

ChordDiagram.prototype.genMatrix = function(threshold){
    this.display_matrix=jQuery.extend(true,[],this.matrix);
    this.display_array=jQuery.extend(true,[],this.index_array);
    var n=this.display_matrix.length;
    var i,j,temp_array=[];
    var sum_in = new Array(n);
    var sum_out = new Array(n);
    for (i=0;i<n;i++){
        sum_in[i]=0;
        sum_out[i]=0;
    }
    for (i=0;i<n;i++)
        for (j=0;j<n;j++){
            sum_in[i]+=this.display_matrix[i][j];
            sum_out[i]+=this.display_matrix[j][i];
        }
    for (i=0;i<n;i++)
        if (sum_in[i]<threshold&&sum_out[i]<threshold) this.display_array[i]=undefined;

    for (i=n-1;i>=0;--i)
        if (this.display_array[i]==undefined) {
            this.display_matrix.splice(i,1);
        } else  {
            temp_array.unshift(this.display_array[i]);
        }


    for (i=0;i<temp_array.length;i++)
        for (j=n;j>0;--j){
            if (this.display_array[j]==undefined) {
                this.display_matrix[i].splice(j,1);
            }
        }

    this.display_array = temp_array;

    n = this.display_array.length;
    if (n<=0){
        console.log("no data satisfy the threshold to draw diagram");
        return;
    }
    this.angle_array = new Array(n);
    //this.cnt_x = this.display_array[0].cfg.cnt_x;
    //this.cnt_y = this.display_array[0].cfg.cnt_y;
    for (i=0;i<n;i++){
        if (this.display_array[i].cfg.cnt_y-this.cnt_y>0) {
            this.angle_array[i] = (this.display_array[i].cfg.cnt_x-this.cnt_x)
                /(this.display_array[i].cfg.cnt_y-this.cnt_y);
            this.angle_array[i] = Math.atan(this.angle_array[i]);
            this.angle_array[i] = this.angle_array[i]>0? this.angle_array[i]:(this.angle_array[i]+Math.PI*2);
        } else if (this.display_array[i].cfg.cnt_y-this.cnt_y<0) {
            this.angle_array[i] = (this.display_array[i].cfg.cnt_x-this.cnt_x)
                /(this.display_array[i].cfg.cnt_y-this.cnt_y);
            this.angle_array[i] = Math.atan(this.angle_array[i]);
            this.angle_array[i] = this.angle_array[i]+Math.PI;
        } else if (this.display_array[i].cfg.cnt_y-this.cnt_y==0) {//though it usually not happens
            if (this.display_array[i].cnt_x-this.cnt_x>=0) this.angle_array[i] = Math.PI/2.0; else
                this.angle_array[i] = Math.PI*3.0/2.0;
        }
    }

    var sort_matrix=this.display_matrix;
    var sort_array=this.angle_array;
    //change the order of data in matrix

    var sum = new Array(n);
    for (i=0;i<n;i++){
        sum[i]=0;
        for (j=0;j<n;j++){
            sum[i]+=sort_matrix[i][j];
            sum[i]+=sort_matrix[j][i];
        }
    }
    var total = 0;
    for (i=0;i<n;i++) total+=sum[i];
    for (i=0;i<n;i++){
        sort_array[i]-=sum[i]*Math.PI/(total*2);
    }

   quickSort(sort_array, function(a,b){return a-b}, 0, sort_array.length-1, sort_matrix);

   function swap(arr,from,to,matrix){
       if (from == to) return;
       var temp = arr[from];
       arr[from] = arr[to];
       arr[to] = temp;
       for (var i=0;i<matrix.length;i++){
           temp = matrix[i][from];
           matrix[i][from] = matrix[i][to];
           matrix[i][to] = temp;
       }
       temp = matrix[from];
       matrix[from] = matrix[to];
       matrix[to] = temp;
   }

   function quickSort(arr, func, from ,to, matrix){
       if (!arr || !arr.length) return [];
       if (from >= to ) return arr;
       var pivot = arr[from];
       var smallEnd = from+1;
       var bigBegin = to;
       while (smallEnd < bigBegin) {
           while (func(arr[bigBegin], pivot) > 0 && smallEnd < bigBegin) { bigBegin--; }
           while (func(arr[smallEnd], pivot) < 0 && smallEnd < bigBegin) { smallEnd++; }
           if (smallEnd < bigBegin) {
               swap(arr, smallEnd, bigBegin, matrix);
           }
       }
       if (func(arr[smallEnd], pivot) < 0) swap(arr, smallEnd, from, matrix);
       quickSort(arr, func, from, smallEnd - 1, matrix);
       quickSort(arr, func, smallEnd, to, matrix);
   }

   this.matrix = sort_matrix;
   this.index_array = sort_array;
};