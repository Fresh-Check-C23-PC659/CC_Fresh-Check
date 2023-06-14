# import requests

# # resp = requests.post("http://127.0.0.1:5000", files={'file': open('fresh_capsicum.png', 'rb')})
# resp = requests.post("https://predictfruit-73izf5446q-as.a.run.app", files={'file': open('fresh_capsicum.png', 'rb')})

# # response_data = resp.json()
# # print(response_data)

# try:
#     response_data = resp.json()
#     print(response_data)
# except requests.exceptions.JSONDecodeError as e:
#     print("Error decoding JSON response:", e)

# # Print the response content
# print(resp.text)

import requests

resp = requests.post("https://freshcheck-ibtomqlr6q-de.a.run.app", files={'file': open('stale_orange.png', 'rb')})

if resp.status_code == 200:
    try:
        response_data = resp.json()
        print(response_data)
    except requests.exceptions.JSONDecodeError as e:
        print("Error decoding JSON response:", e)
else:
    print("Error:", resp.status_code, resp.text)

