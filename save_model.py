from transformers import GPTNeoForCausalLM, GPT2Tokenizer, pipeline

def generate_and_save_model(title):
    # Set up the text generation pipeline
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')

    # Generate essay based on the title
    result = generator(title, max_length=300, do_sample=True, temperature=0.9)
    generated_text = result[0]['generated_text']

    # Save the generated text to a file (optional)
    with open('./generated_essay.txt', 'w', encoding='utf-8') as file:
        file.write(generated_text)

    # Save the model and tokenizer used for generation
    model_name = 'EleutherAI/gpt-neo-1.3B'
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    save_directory = './gpt-neo-1.3B-model'
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    print("Model and tokenizer saved successfully.")

if __name__ == "_main_":
    # Example usage: generate an essay based on a title and save the model and tokenizer
    title = "The Impact of Artificial Intelligence on Society"
    generate_and_save_model(title)