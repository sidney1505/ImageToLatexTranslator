var Svg = require('svgutils').Svg;
var fs = require('fs');
 
Svg.fromSvgDocument("/home/sbender/Documents/sample.svg", function(err, svg){
    if(err){
        throw new Error('SVG file not found or invalid');
    } 
    var json = svg.toJSON();
    fs.writeFile("/home/sbender/Documents/sample.json", JSON.stringify(json), function(err) {
	    if(err) {
	        return console.log(err);
	    }
	});
});