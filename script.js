const temperatureInput = document.querySelector(".input-temperature");
const resultInput = document.querySelector(".output-temperature");
const inputUnit = document.querySelector(".input-unit");
const outputUnit = document.querySelector(".output-unit");

temperatureInput.addEventListener("input", convertTemperature);
inputUnit.addEventListener("change", convertTemperature);
outputUnit.addEventListener("change", convertTemperature);

function convertTemperature() {
    const inputValue = parseFloat(temperatureInput.value);
    if (isNaN(inputValue)) {
        resultInput.value = "";
        return;
    }

    const inputUnitValue = inputUnit.value;
    const outputUnitValue = outputUnit.value;
    let convertedTemperature = 0;

    if (inputUnitValue === "Celsius") {
        if (outputUnitValue === "Kelvin") {
            convertedTemperature = inputValue + 273.15;
        } else if (outputUnitValue === "Fahrenheit") {
            convertedTemperature = (inputValue * 9/5) + 32;
        } else {
            convertedTemperature = inputValue;
        }
    } else if (inputUnitValue === "Kelvin") {
        if (outputUnitValue === "Celsius") {
            convertedTemperature = inputValue - 273.15;
        } else if (outputUnitValue === "Fahrenheit") {
            convertedTemperature = (inputValue - 273.15) * 9/5 + 32;
        } else {
            convertedTemperature = inputValue;
        }
    } else if (inputUnitValue === "Fahrenheit") {
        if (outputUnitValue === "Celsius") {
            convertedTemperature = (inputValue - 32) * 5/9;
        } else if (outputUnitValue === "Kelvin") {
            convertedTemperature = (inputValue - 32) * 5/9 + 273.15;
        } else {
            convertedTemperature = inputValue;
        }
    }

    resultInput.value = convertedTemperature.toFixed(2);
}
