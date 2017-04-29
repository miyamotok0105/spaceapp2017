var express = require('express');
var app = express();


/* 
実行
node app.js

exsample site
https://bl.ocks.org/mbostock/899711

if use this , first input api key like hogehoge.
<script src="https://maps.googleapis.com/maps/api/js?key=hogehoge"></script>

http://localhost:3000/examples/mapd3/index.html

*/

app.use('/examples', express.static(__dirname + '/examples'));


app.get('/', function (req, res) {
  res.send('Hello World!');
});

app.listen(3000, function () {
  console.log('Example app listening on port 3000!');
});

