from django.shortcuts import render, redirect
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
import pickle,datetime,requests
from .forms import CityForm 
import random
from django.shortcuts import render
from django.core.files.storage import default_storage
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing import image
import numpy as np
city=''
sensor_data = {
        'state': 'California',
        'maxTemp': 30,
        'minTemp': 20,
        'rhi': 70,
        'rainfall_data': 10,
        'rhii': 80,
        'sunshine_hours': 6
    }
# Define the custom layer 'FixedDropout'
class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)
from django.shortcuts import render

# print(sensor_data.state)

# Function for generating predictions
def getPredictions(file_url):
    # Load your trained CNN model
    model = load_model('efficient_net_model.h5', custom_objects={'FixedDropout': FixedDropout})

    # Load and preprocess the image
    img = image.load_img(file_url, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Perform prediction
    prediction = model.predict(img_array)

    # Process prediction result (you may need to adjust this part based on your model output)
    # For example, if your model outputs a probability distribution, you might want to get the class with the highest probability
    predicted_class_index = np.argmax(prediction)
    class_names = ["Yellow Stem borer", "Green Leaf Hopper", "Brown Plant Hopper"]  # Replace with your class names
    predicted_class = class_names[predicted_class_index]

    # Constructing the path to the image (you may need to adjust this based on your project structure)
    image_path = "/media/{}".format(file_url.split('/')[-1])

    # Returning the prediction result and the path to the image
    return predicted_class, image_path
import json
# Upload image view
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        file_name = default_storage.save(uploaded_image.name, uploaded_image)
        file_url = default_storage.path(file_name)

        # Generate predictions
        prediction_result= getPredictions(file_url)

        # Render result page
        return render(request, 'sensor_pre.html', {'your_value': prediction_result,'sensor_data':sensor_data})
    else:
        return render(request, 'index.html')

from django.http import JsonResponse
def get_cities(request):
    state = request.GET.get('state')
    cities = STATE_CITIES.get(state, [])
    return JsonResponse(cities, safe=False)
@csrf_exempt
# get data from form and add to db

def home(request, city=None, state=None):
    template = loader.get_template('index.html')
    prediction = ''
    region = ''
    print(request,city,state)
    if city and state :

        region = f"{city.capitalize()}, {state.capitalize()}"
        prediction=getWeatherData(city,state)
        return render(request, 'sensor_pre.html', {'your_value': prediction, 'your_region': region})
            
    else:
        form = CityForm()
    return HttpResponse(template.render({'form': form, 'prediction': prediction, 'region': region}, request))

def getWeatherData(city, state):
    weather_api_key = '4b69e9c09afabe33e9c0aa775a8400ce'

    # First, fetch current weather data
    current_weather_url = f'http://api.openweathermap.org/data/2.5/weather?q={city},{state}&appid={weather_api_key}'
    current_weather_response = requests.get(current_weather_url).json()

    # Then, fetch forecast data to get rainfall information
    forecast_url = f'http://api.openweathermap.org/data/2.5/forecast?q={city},{state}&appid={weather_api_key}'
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
    rainfall_data = forecast_response['list'][0]['rain']['3h'] if 'rain' in forecast_response['list'][0] else 0  # Rainfall in the last 3 hours (in mm)

    pickle_file_path = 'gbclassifier.pickle'
    try:
            # Load the pickled model from the file
            with open(pickle_file_path, 'rb') as file:
                sensors_model = pickle.load(file)
    except FileNotFoundError:
            print(f"Error: File '{pickle_file_path}' not found.")
    except Exception as e:
            print("Error loading the pickled model:", e)

    state_encoding={"TamilNadu":5,"Chhattisgarh":2,"Telangana":3,"WestBengal":1,"Karnataka":4}
    rhii = random.randint(50, 90)
    sensors_data = [
        [state_encoding[state], maxTemp, minTemp, rhi, rainfall_data, rhii, sunshine_hours]
    ]
    print(sensors_data)


    # Makeing prediction using the loaded model
    prediction = sensors_model.predict(sensors_data)
    prediction = prediction.astype(int)
    print("Prediction:", prediction)

    pest = ""
    if prediction == 1:
        pest = "Green Leaf Hopper"
    elif prediction == 2:
        pest = "White Blacked Plant Hopper"
    elif prediction == 3:
        pest = "Yellow Stem borer"
    return pest


STATE_CITIES = {
    'WestBengal': ['Howrah', 'Durgapur', 'Asansol', 'Siliguri', 'Maheshtala',
                  'Rajarhat Gopalpur', 'Bardhaman', 'Kharagpur', 'Halisahar', 'Bally',
                  'Barasat', 'Krishnanagar', 'Baharampur', 'Habra', 'Santipur',
                  'Bankura', 'Balurghat', 'Basirhat', 'Chandannagar', 'Cooch Behar',
                  'Alipurduar', 'Purulia', 'Jalpaiguri', 'Kalimpong', 'Madhyamgram'],
    'Chhattisgarh': ['Durg', 'Bhilai', 'Rajnandgaon', 'Bilaspur', 'Korba',
               'Ambikapur', 'Jagdalpur', 'Champa', 'Janjgir', 'Raigarh',
               'Mahasamund', 'Kawardha', 'Bhatapara', 'Sakti', 'Tilda',
               'Baloda Bazar', 'Dhamtari', 'Mungeli', 'Saraipali', 'Bemetara'],
    'Telangana': ['Badangpet', 'Bellampalle', 'Bhadrachalam', 'Bhuvanagiri', 'Bodhan',
                  'Gadwal', 'Jangaon', 'Kagaznagar', 'Kamareddy', 'Khanapuram Haveli',
                  'Kodad', 'Korutla', 'Kothagudem', 'Mandamarri', 'Meerpet–Jillelguda',
                  'Metpally', 'Nirmal', 'Palwancha', 'Peerzadiguda', 'Sangareddy',
                  'Sircilla', 'Tandur', 'Vikarabad', 'Wanaparthy', 'Zahirabad'],
    'Karnataka': ['Hubli', 'Mangalore', 'Gulbarga', 'Belgaum', 'Davanagere',
               'Bellary', 'Bijapur', 'Shimoga', 'Tumkur', 'Raichur',
               'Bidar', 'Hospet', 'Hassan', 'Udupi', 'Gadag',
               'Robertson Pet', 'Bhadravati', 'Chitradurga', 'Kolar', 'Mandya',
               'Chikmagalur', 'Gangawati', 'Bagalkot', 'Ranebennuru', 'Sirsi',
               'Karwar', 'Chamarajanagar'],
    'TamilNadu': ['Coimbatore', 'Madurai', 'Tiruchirappalli', 'Tiruppur', 'Salem',
                  'Erode', 'Tirunelveli', 'Vellore', 'Thoothukkudi', 'Thanjavur',
                  'Dindigul', 'Ranipet', 'Sivakasi', 'Karur', 'Ooty',
                  'Hosur', 'Nagercoil', 'Kanchipuram', 'Kumbakonam', 'Tiruvannamalai']
}


