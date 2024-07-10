from flask import Flask, request, jsonify, render_template
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the model and tokenizer
model = GPTNeoForCausalLM.from_pretrained('./gpt-neo-1.3B-model')
tokenizer = GPT2Tokenizer.from_pretrained('./gpt-neo-1.3B-model')

@app.route('/')
def home():
    logging.info("Rendering home page.")
    return render_template('index.html')
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 500)
    
    logging.info(f"Received prompt: {prompt}")
    
    try:
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        logging.info("Tokenized input successfully.")
        
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
        logging.info("Generated output successfully.")
        
        essay = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info("Decoded output successfully.")
        
        return jsonify({'essay': essay})
    except Exception as e:
        logging.error(f"Error during essay generation: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    logging.info("Starting Flask app...")
    app.run(debug=True)






# from flask import Flask, request, jsonify, render_template

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/generate', methods=['POST'])
# def generate():
#     data = request.get_json()
#     prompt = data.get('prompt', '')
#     max_length = data.get('max_length', 500)
    
#     # Mock response for testing purposes
#     essay = f"This is a mock essay generated for the topic: {prompt}. This essay is a placeholder to test the UI functionality."
    
#     return jsonify({'essay': essay})

# if __name__ == '__main__':
#     app.run(debug=True)


#  Current exist
# from flask import Flask, request, jsonify, render_template
# import openai
# import os
# from dotenv import load_dotenv

# load_dotenv()  # Load environment variables from .env file

# app = Flask(__name__)

# # Set the OpenAI API key
# openai.api_key = os.getenv("OPENAI_API_KEY")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/generate', methods=['POST'])
# def generate():
#     data = request.get_json()
#     prompt = data.get('prompt', '')
#     max_tokens = data.get('max_tokens', 500)

#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": f"Write an essay about {prompt}"}
#             ],
#             max_tokens=max_tokens,
#             temperature=0.7,
#             top_p=0.95,
#             n=1
#         )
#         essay = response.choices[0].message['content'].strip()
#         return jsonify({'essay': essay})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


#from flask import Flask, request, jsonify, render_template
#import requests
#import os
#from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

#app = Flask(__name__)

# # Get the Claude API key from environment variables
#api_key = os.getenv('CLAUDE_API_KEY')
#if not api_key:
#  raise ValueError("No Claude API key found. Please set the CLAUDE_API_KEY in your .env file.")

#@app.route('/')
#def home():
#    return render_template('index.html')

#@app.route('/generate', methods=['POST'])
#def generate():
#    data = request.get_json()
#    prompt = data.get('prompt', '')
 #   max_length = data.get('max_length', 500)
    
 #   full_prompt = f"Write an essay about {prompt}"
    
#     # Make a request to Claude API
#    headers = {
#        'Authorization': f'Bearer {api_key}',
#         'Content-Type': 'application/json',
#     }
#    payload = {
#         'prompt': full_prompt,
#         'max_tokens': max_length,
#         'temperature': 0.7,
#     }
#    response = requests.post('https://api.anthropic.com/v1/complete', headers=headers, json=payload)
    
#    if response.status_code == 200:
#         result = response.json()
#         essay = result['choices'][0]['text'].strip()
#    else:
#         essay = f"Error: {response.status_code}, {response.text}"
    
#     return jsonify({'essay': essay})

#if __name__ == '__main__':
#     app.run(debug=True)












# from flask import Flask, request, jsonify, render_template
# from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel

# app = Flask(__name__)

# # Set up the text generation pipeline
# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/generate', methods=['POST'])
# def generate():
#     data = request.get_json()
#     prompt = data.get('prompt', '')
#     max_length = data.get('max_length', 500)
    
#     try:
#         # Generate the essay based on the prompt provided by the user
#         result = generator(prompt, max_length=max_length, do_sample=True, temperature=0.9)
#         essay = result[0]['generated_text']
#         return jsonify({'essay': essay})
#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)