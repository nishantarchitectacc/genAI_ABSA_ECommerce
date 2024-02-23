const express = require('express');
const app = express();
const port = 3000;
const path = require('path');
const cor = require('cors')
var bodyParser = require('body-parser')
const http = require('http');
const axios = require('axios');



app.use(function (req, res, next) {
    res.header("Access-Control-Allow-Origin", "*"); // update to match the domain you will make the request from
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
});



// const getPosts = (inputFeedback) => {
//     console.log('inputFeedback', inputFeedback);
//   let data = '';
//     app.post('http://127.0.0.1:5000', (response) => {
//         response.setEncoding('utf8');

//         console.log('response', response);
//         console.log('response status', response.statusCode);

//         // response.header("Access-Control-Allow-Origin", "*"); // update to match the domain you will make the request from
//         // response.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
//         response.on('data', (chunk) => {
//             data += chunk;
//         });

//     response.on('end', () => {
//         console.log('data', data);
//       console.log(data);
//     });
//   });

// //   request.on('error', (error) => {
// //     console.error(error);
// //   });

// //   request.end();
// };

app.use(cor())
app.use(bodyParser.json())
app.post('/', (req, res) => {
    // console.log('res', res);
    console.log('req', req.body);
    let url = `http://127.0.0.1:5000/api/ragllama2`;
    // res.send(data);


    axios({
        method: 'post',
        url,
        data: {
            user_message: req.body.value
        },
        headers: { "Content-Type": "application/json" }
    })
        .then(function (response) {
            console.log('response', response);
            res.send(JSON.stringify(response.data));

        })
        .catch(function (error) {
            console.log('aios error message', error);
        });
});

// app.use(express.static(path.join(__dirname, '../dist/material-dashboard-angular2-master')));

// app.get('*', (req, res) => {
//   res.sendFile(path.join(__dirname, '../dist/material-dashboard-angular2-master/index.html'));
// });

app.listen(port, () => {
    console.log(`Server listening at http://localhost:${port}`);
});
