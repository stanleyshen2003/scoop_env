import requests
response = requests.post('https://fumes-api.onrender.com/llama3',
 json={
 'prompt': """{
   
 'systemPrompt': 'Reply in a single action in action list [scoop, move, fork]', 
 'user': 'I want to eat the beans in bowl'
  
 }""",
 "temperature":0.75,
 "topP":0.9,
 "maxTokens": 1

}, stream=True)
for chunk in response.iter_content(chunk_size=1024):  
 if chunk:
      print(chunk.decode('utf-8'))