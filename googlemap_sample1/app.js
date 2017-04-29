var express = require('express');
var app = express();


/* 
実行
node app.js

exsample site
https://bl.ocks.org/mbostock/899711

http://localhost:3000/gmaps/examples/mapd3/index.html

*/

app.use('/gmaps', express.static(__dirname + '/gmaps'));


app.get('/', function (req, res) {
  res.send('Hello World!');
});

app.listen(3000, function () {
  console.log('Example app listening on port 3000!');
});

