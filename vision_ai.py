

import requests

# Set the endpoint URL
url = 'https://sinkin.ai/api/inference'


Room = "Exterior"
color_them = "Navy Blue and White"
Design = "Indestrial"
Day_time = "Sunrise"
Lighting_Style = "Natural"
prompt = (
    f"Give me a realistic and complete image of"
    f"room type: {Room}, Color Theme : {color_them}, Lightining style : {Lighting_Style}"
    f"Day time : {Day_time} and Design style: {Design}"
    )
# Set the required parameters
params = {
	'access_token': '3e03eaedae3846f2bd3edc5167d25fcc',
	'model_id': '4zdwGOB',
	'prompt': prompt,
	'num_images': 1,
	'scale': 7,
	'steps': 30,
	'width': 896,
	'height': 896,
}

# Set the optional parameters
files = {'init_image_file': open('exterior.jpg', 'rb')}  # Add your image file here

# Send the POST request
response = requests.post(url, files=files, data=params)

# Check the response status code
if response.status_code == 200:
	# Parse the response JSON
	data = response.json()

	# Check if the API request was successful
	if data['error_code'] == 0:
		# Access the resulting images
		images = data['images']
		for image_url in images:
			print(f"Inference image URL: {image_url}")
	else:
		# Handle the API error
		error_message = data['message']
		print(f"API request failed: {error_message}")
else:
	# Handle the HTTP request error
	print(f"HTTP request failed with status code: {response.status_code}")
