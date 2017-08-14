const fs = require("pn/fs"); // https://www.npmjs.com/package/pn
var svg_to_png = require('svg-to-png');
 
svg_to_png.convert("/home/sbender/Documents/sample.svg", "/home/sbender/Documents/sample2.png") // async, returns promise 
.then( function(){
    // Do tons of stuff 
});


/*const svg2png = require("svg2png");

fs.readFile("/home/sbender/Documents/sample.svg")
svg2png(sourceBuffer, {url: "/home/sbender/Documents/sample.svg", width: 500, height: 500 })
    .then(buffer => fs.writeFile("/home/sbender/Documents/sample.png", buffer))
    .catch(e => console.error(e));*/