from django.shortcuts import render, redirect
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
import pickle,datetime,requests
from .forms import CityForm 
city=''

@csrf_exempt
# get data from form and add to db
def home(request):
    template = loader.get_template('index.html')
    prediction=''
    if request.method == 'POST':
        form = CityForm(request.POST)
        if form.is_valid():
            city= form.cleaned_data['city']
            prediction=getWeatherData(city)
            city=city[0].upper()+city[1:]
            return render(request, 'sensor_pre.html', {'your_value': prediction,'your_region':city})
    else:
        form = CityForm()
    
    return HttpResponse(template.render({'form': form,'prediction':prediction}, request))

def getWeatherData(city):
    weather_api_key = '4b69e9c09afabe33e9c0aa775a8400ce'

# First, fetch current weather data
    current_weather_url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}'
    current_weather_response = requests.get(current_weather_url).json()

    # Then, fetch forecast data to get rainfall information
    forecast_url = f'http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={weather_api_key}'
    forecast_response = requests.get(forecast_url).json()

    # Extract relevant information
    minTemp = current_weather_response['main']['temp_min'] - 273.15  # Convert to Celsius
    maxTemp = current_weather_response['main']['temp_max'] - 273.15  # Convert to Celsius
    rhi = current_weather_response['main']['humidity']
    data = current_weather_response['sys'] # Sunshine hours in the last 3 hours
    sunrise_time = datetime.datetime.utcfromtimestamp(data['sunrise'])
    sunset_time = datetime.datetime.utcfromtimestamp(data['sunset'])
    time_difference_seconds = (sunset_time - sunrise_time).total_seconds()

    sunshine_hours = time_difference_seconds / 3600
    # Extract rainfall data from the forecast
    rainfall_data = forecast_response['list'][0]['rain']['24h'] if 'rain' in forecast_response['list'][0] else 0  # Rainfall in the last 3 hours (in mm)

    sensors_data = [
        [maxTemp, minTemp, rhi, rainfall_data, sunshine_hours]]
    sensors_model = pickle.load(open('randomForestFinal.pkl', 'rb'))
    prediction = sensors_model.predict(
        sensors_data)
    prediction= prediction.astype(int)
    
    pest=""
    if prediction == 1:
        pest="Green Leaf Hopper"
    elif prediction == 2:
        pest="White Blacked Plant Hopper"
    elif prediction == 3:
        pest="Brown Plant Hopper"
    elif prediction == 4:
        pest="Leaf Folder"
    elif prediction == 5:
        pest="Yellow Stem borer"
    return pest



