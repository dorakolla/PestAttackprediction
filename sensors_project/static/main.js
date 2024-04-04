 document.addEventListener('DOMContentLoaded', function() {
 document.getElementById('weather-button').addEventListener('click', function() {
 console.log("kjgbwrk")
 var state = "State: " + state_encoding[state];
 var maxTemp = "Max Temperature: " + maxTemp + "°C";
 var minTemp = "Min Temperature: " + minTemp + "°C";
 var rhi = "RHI: " + rhi;
 var rainfall = "Rainfall: " + rainfall_data + "mm";
 var rhii = "RHII: " + rhii;
 var sunshine = "Sunshine Hours: " + sunshine_hours;
 // Construct the details string
 var details = state + "<br>" + maxTemp + "<br>" + minTemp + "<br>" + rhi + "<br>" + rainfall + "<br>" + rhii + "<br>" + sunshine;
 // Display the details
 document.getElementById('weather-details').innerHTML = details;
 document.getElementById('weather-details').style.display = 'block';
 })
                 });

    document.addEventListener('DOMContentLoaded', function() {
        var stateSelect = document.getElementById('id_state');
        var citySelect = document.getElementById('id_city');
        var submitButton = document.getElementById('submit-btn');
        var stateValue = stateSelect.value;

        // Add a default option for city select
        var defaultOption = document.createElement('option');
        defaultOption.text = 'Select a city';
        defaultOption.value = '';
        citySelect.appendChild(defaultOption);
        citySelect.selectedIndex = 0; // Set default option as selected

        stateSelect.addEventListener('change', function() {
            var stateValue = stateSelect.value;
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '/get-cities/?state=' + stateValue, true);
            xhr.onload = function() {
                if (xhr.status === 200) {
                    citySelect.innerHTML = ''; // Clear existing options
                    citySelect.appendChild(defaultOption); // Add default option back
                    var cities = JSON.parse(xhr.responseText);
                    cities.forEach(function(city) {
                        var option = document.createElement('option');
                        option.text = city;
                        option.value = city;
                        citySelect.appendChild(option);
                    });
                } else {
                    console.error('Request failed. Status:', xhr.status);
                }
            };
            xhr.onerror = function() {
                console.error('Request failed. Network error.');
            };
            xhr.send();
        });

        submitButton.addEventListener('click', function() {
            var cityValue = citySelect.value;
            var stateValue = stateSelect.value;
            var form = document.getElementById('weather-form');
            form.action = '/home/' + cityValue + '/' + stateValue + '/';
            form.submit();
        });
    });

    document.addEventListener('DOMContentLoaded', function() {
        var stateSelect = document.getElementById('id_state');
        var citySelect = document.getElementById('id_city');
        var submitButton = document.getElementById('submit-btn');
        var stateValue = stateSelect.value;

        // Add a default option for city select
        var defaultOption = document.createElement('option');
        defaultOption.text = 'Select a city';
        defaultOption.value = '';
        citySelect.appendChild(defaultOption);
        citySelect.selectedIndex = 0; // Set default option as selected

        stateSelect.addEventListener('change', function() {
            var stateValue = stateSelect.value;
            var xhr = new XMLHttpRequest();
            xhr.open('GET', '/get-cities/?state=' + stateValue, true);
            xhr.onload = function() {
                if (xhr.status === 200) {
                    citySelect.innerHTML = ''; // Clear existing options
                    citySelect.appendChild(defaultOption); // Add default option back
                    var cities = JSON.parse(xhr.responseText);
                    cities.forEach(function(city) {
                        var option = document.createElement('option');
                        option.text = city;
                        option.value = city;
                        citySelect.appendChild(option);
                    });
                } else {
                    console.error('Request failed. Status:', xhr.status);
                }
            };
            xhr.onerror = function() {
                console.error('Request failed. Network error.');
            };
            xhr.send();
        });

        submitButton.addEventListener('click', function() {
            var cityValue = citySelect.value;
            var stateValue = stateSelect.value;
            var form = document.getElementById('weather-form');
            form.action = '/home/' + cityValue + '/' + stateValue + '/';
            form.submit();
        });
    });