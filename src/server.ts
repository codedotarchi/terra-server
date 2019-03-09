

// let tf = require('@tensorflow/tfjs');
tf = require('@tensorflow/tfjs-node-gpu');

console.log("Hello World");

function helloWorld() {
    return console.log('Hello World from a function');
}


let t = tf.ones([128, 128]);
console.log(t);
