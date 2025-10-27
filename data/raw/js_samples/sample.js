import { otherFunc } from './other.js';

const PI = 3.14;

function greet(name) {
    const message = `Hello, ${name}!`;
    console.log(message);
    otherFunc();
    return message;
}

export default greet;
