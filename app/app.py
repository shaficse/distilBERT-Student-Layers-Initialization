from flask import Flask, request, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

app = Flask(__name__)

# Load models and tokenizer
load_directory = "./models"
tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_directory, "tokenizer"))
teacher_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(load_directory, "teacher_model"))
student_models = {
    "top_k": AutoModelForSequenceClassification.from_pretrained(os.path.join(load_directory, "student_model_top_k")),
    "bottom_k": AutoModelForSequenceClassification.from_pretrained(os.path.join(load_directory, "student_model_bottom_k")),
    "odd_layers": AutoModelForSequenceClassification.from_pretrained(os.path.join(load_directory, "student_model_odd_layers")),
    "even_layers": AutoModelForSequenceClassification.from_pretrained(os.path.join(load_directory, "student_model_even_layers"))
}

# Define id2label mapping
id2label = {0: "Label1", 1: "Label2"}  # Update based on your actual labels

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to device
teacher_model = teacher_model.to(device)
for model in student_models.values():
    model.to(device)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Perform inference and collect predictions
        predictions = {}

        teacher_model.eval()
        with torch.no_grad():
            outputs = teacher_model(**inputs)
            predictions['Teacher'] = teacher_model.config.id2label[torch.argmax(outputs.logits, dim=-1).item()]

        for strategy, model in student_models.items():
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                predictions[strategy] = model.config.id2label[torch.argmax(outputs.logits, dim=-1).item()]

        return render_template('index.html', predictions=predictions, input_text=input_text)
    return render_template('index.html', predictions={})

if __name__ == '__main__':
    app.run(debug=True)
