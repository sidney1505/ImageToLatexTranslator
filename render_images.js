
/*var fs = require('fs');
var readline = require('readline');
var rd = readline.createInterface({
    input: fs.createReadStream(process.env.FORMULAS),
    output: process.stdout,
    console: false
});


rd.on('line', function(line) {
  console.log(line);  
  options = {math: line, format: "TeX", svg:true,}
  callback = function(data) {
    console.log('callback')
    if (!data.errors) {
      console.log('||||||||||||||||||||||||||||||||||||||||||||')
      console.log('Result svg:')
      console.log(data.svg)
      console.log('||||||||||||||||||||||||||||||||||||||||||||')
    }
  }
  callback(line);
  x = mjAPI.typeset(options, callback);
  console.log(x)
  console.log('______________________________')
});*/

var mjAPI = require("mathjax-node");
mjAPI.config({MathJax: {}});
mjAPI.start();

var lineReader = require('line-reader');
var fs = require('fs');
var i = 0;
lineReader.eachLine(process.env.FORMULAS, function(line, last) {
  if(i % 100 == 0) {
    console.log(i);
  }
  options = {math: line, format: "TeX", svg:true,}
  callback = function(data) {
    if(i % 100 == 0) {
      console.log(line);
      console.log('::::::::');
      console.log(data.svg);
      console.log('::::::::');
    }
    if (!data.errors) {
      fs.writeFile(process.env.SVG_IMAGES+'/img'+i.toString()+'.svg',data.svg,function(err) {
        if(err) {
            return console.log(err);
        } else if(i % 100 == 0) {
          console.log('img' + i.toString() + ' saved!')
        }
      });
    }
  }
  require("mathjax-node").typeset(options, callback);
  i++;
});


/*
var mjAPI = require("mathjax-node");
mjAPI.config({MathJax: {}});
mjAPI.start();
var options = {math: '{ \\theta _ { n } ^ { \\alpha \\Lambda } } { ^ \\dagger } = \\theta _ { - n } ^ { \\alpha \\Lambda } ,', format: "TeX", svg:true,}
var callback = function(data) {console.log('callback'); console.log(data.svg);}
mjAPI.typeset(options, callback);
*/