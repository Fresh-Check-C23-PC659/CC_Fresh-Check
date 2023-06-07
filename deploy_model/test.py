import requests

# resp = requests.post("http://127.0.0.1:5000", files={'file': open('stale_tomato.jpeg', 'rb')})
resp = requests.post("https://freshcheck-ibtomqlr6q-de.a.run.app", files={'file': open('test_image/stale_tomato.jpeg', 'rb')})

try:
    response_data = resp.json()
    print(response_data)
except requests.exceptions.JSONDecodeError as e:
    print("Error decoding JSON response:", e)

# import requests

# resp = requests.post("https://getprediction-73izf5446q-as.a.run.app", files={'file': open('stale_tomato.jpeg', 'rb')})

# # Print the response status code
# print("Response status code:", resp.status_code)

# # Check if the response is successful (status code 200)
# if resp.status_code == 200:
#     try:
#         response_data = resp.json()
#         print(response_data)
#     except requests.exceptions.JSONDecodeError as e:
#         print("Error decoding JSON response:", e)
# else:
#     print("Error: Response status code", resp.status_code)
