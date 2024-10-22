import gradio as gr
from transformers import pipeline

get_completion = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def model(input):
    output = get_completion(input)
    return output[0]['summary_text']

demo = gr.Interface(fn=model,
                    inputs=[gr.Textbox(label="Text to summerize")],
                    outputs=[gr.Textbox(label="summary")],
                    examples=["The tower is 324 meters (1,063 ft) tall, about the same height as an 81-story building, and the tallest structure in Paris. Its base is square, measuring 125 meters on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 meters. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 meters (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."],
                    example_labels=["Eiffel Tower"],
                    title="Summarization App",
                    description="This app summarizes text using a distilbart-cnn-12-6 model.")


demo.launch(share=True)

