import openai 
import os
# set openai key

def get_api_key(txt_file):
    contents = ''
    with open(txt_file) as f:
        contents = f.read()
    return contents

openai.api_key  = get_api_key('gpt_key.txt')

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

problem = """
Evaluator: ['a cat laying on top of a pile of clothes', 'a row of boats lined up in a row', 'a white and blue boat on the water', 'a cartoon of a bird flying through the air', 'a small black and white bird standing on top of a green field', 'a large brown bear sitting on top of a dirt ground', 'a car that is sitting in the middle of a room', 'a green frog is standing on top of a table', 'a cat that is laying down on a couch', 'a black car is driving down the street', 'a large white bird flying through the air', 'a large white truck driving down a street', 'a person sitting on a couch with a dog', 'a white horse standing on top of a lush green field', 'a blue truck parked on the side of a road', 'a boat in the middle of a body of water', 'a brown and white dog holding a ball', 'a brown and white horse standing next to a fence', 'a large boat in the middle of a body of water', 'a black and white photo of a brown and white bird', 'a brown horse standing next to a white horse', 'a black and white photo of a black and white bird flying', 'a large black and white bird standing on top of a body of water', 'a red fire truck parked in front of a building', 'a small dog standing in a grassy field', 'a man standing next to a black and white bird', 'a herd of zebra standing next to each other', 'a model of a fighter jet flying through the air', 'a truck is parked on the side of the road', 'a black and white photo of a brown and white bird', 'a cat laying on the ground next to a dog', 'a dog laying in the grass']

Client 68:['a large brown and white cow standing next to a wooden fence', 'a black and white dog standing next to a white and brown dog', 'a large bird standing on top of a lush green field', 'a painting of a person standing on top of a yellow fire hydrant', 'a person riding a horse in a field', 'a white truck is driving down the road', 'a small brown and white bird standing in the middle of a forest', 'a small dog is playing with a toy in a room', 'a white dog sitting in front of a christmas tree', 'a large building with a train on top of it']     
Client 66:['a large jetliner flying through a blue sky', 'a plane flying over a body of water', 'a cat standing on top of a lush green field', 'a small bird is standing on the ground', 'a man rowing a boat in the middle of a lake', 'a bird perched on top of a wooden post', 'a car is parked on the side of the road', 'a cat jumping up into the air to catch a frisbee', 'a bird flying over a large body of water', 'a horse that is standing in the dirt']
Client 47:['a large brown and white bird standing in the middle of a field', 'a red and white fire truck parked in front of a building', 'a seagull flying over a blue sky', 'a man riding on the back of a horse', 'a lone zebra standing in the middle of a field', 'a cat that is laying down on the floor', 'a small white dog standing in a grassy field', 'a collage of photos of a man in a suit', 'a car is parked in front of a black and white photo', 'a plane sitting on top of an airport tarmac']
Client 50:['a large brown dog standing next to a tree', 'a black and white dog standing next to a park bench', 'a small white dog sitting in the grass', 'a cat laying on a couch looking at the camera', 'a cat sitting on top of a rock', 'a boat floating on top of a body of water', 'a car that is sitting in the middle of a parking lot', 'a black and white photo of a white car', 'a black and white photo of a black and white bird', 'a cat sitting on top of a pile of clothes']
Client 85:['a brown horse standing next to a white horse', 'a dump truck is parked in a parking lot', 'a large jetliner flying through a grassy field', 'an airplane flying in the sky over a body of water', 'a truck is driving down a dirt road', 'a boat is docked in the middle of the ocean', 'a woman riding on the back of a white horse', "a brown and white horse with a white stripe on it's face", 'a black and white horse standing in a grassy field', 'a black and white dog standing next to a snow covered ground']
Client 30:['a black and white photo of a person holding a pair of scissors', 'a white bird flying through the air', 'a black and white photo of a black and white bird', 'a black and white photo of a white object', 'an old black and white photo of an old black and white photo', 'a black and white photo of a black and white bird', 'a black and white photo of a bird on a white background', 'a black and white photo of a clock', 'a black and white photo of a clock on a black background', 'a black and white photo of a black and white object']
Client 16:['a black and white photo of a black and white bird', 'an old black and white photo of a person holding an american flag', 'a black and white photo of a black and white bird', 'a black and white photo of a black and white bird', 'a black and white photo of a black and white bird', 'a black and white photo of a black and white bird', 'a black and white photo of a black and white bird', 'a black and white photo of a black and white clock', 'a black and white photo of a black and white sign', 'an old black and white photo of an old black and white photo of an']
Client 57:['a black and white photo of a person on a cell phone', 'a black and white photo of a clock on a black and white background', 'a black and white photo of a black and white clock', 'an old black and white photo of a person holding an umbrella', 'a black and white photo of a black and white object', 'a black and white photo of a black and white elephant', 'a large white elephant standing on top of a white cloth', 'an old black and white photo of an old black and white photo of an', 'a white bird sitting on top of a white teddy bear', 'a black and white photo of a black and white clock']
Client 29:['a black and white photo of a black and white bird', 'a black and white photo of a black and white bird', 'a black and white photo of a black and white clock', 'a black and white photo of a bird flying through the air', 'a white bird flying through the air', 'a black and white photo of a black and white bird', 'a black and white photo of a bird on a string', 'an old black and white photo of an american flag', 'a black and white photo of a black and white bird', 'a pair of scissors sitting on top of a table']
"""

prompt = f"""
Your task is to list and sort 3 clients which data description are most similar to the evaluator data description.

The clients and evaluator data descriptions are delimited with triple backticks.
Format your response as a Python list of client ids, sorted from most similar to least similar.

Data description: '''{problem}'''
"""
response = get_completion(prompt)
print(response)

"""

"""