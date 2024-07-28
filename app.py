from flask import Flask, request, jsonify, render_template
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

device = "cuda" if torch.cuda.is_available() else "cpu"

class StrawHatModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

# Ruta al archivo del modelo guardado
model_path = "static/models/StrawHatModel.pth"  # Cambia esto si el archivo está en otro lugar

# Definir el modelo y cargar los pesos guardados
input_shape = 3
hidden_units = 10
output_shape = 5
model = StrawHatModel(input_shape=3, hidden_units=10, output_shape=5)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Lista de clases, imágenes y descripciones
classes = ["luffy", "nami", "sanji", "usopp", "zoro"]
images = {
    "luffy": "static/images/luffy.jpg",
    "nami": "static/images/nami.jpg",
    "sanji": "static/images/sanji.jpg",
    "usopp": "static/images/usopp.jpg",
    "zoro": "static/images/zoro.jpg"
}
descriptions = {
    "luffy": "Luffy is the captain of the Straw Hat Pirates. He has the ability to stretch his body like rubber after eating the Gum-Gum Fruit.",
    "nami": "Nami is the navigator of the Straw Hat Pirates. She is passionate about mapping the entire world.",
    "sanji": "Sanji is the cook of the Straw Hat Pirates. He is known for his culinary skills and chivalry.",
    "usopp": "Usopp is the sniper of the Straw Hat Pirates. He is known for his marksmanship and creative inventions.",
    "zoro": "Zoro is the swordsman of the Straw Hat Pirates. He is known for his use of three swords in battle."
}

# Define a transform to preprocess the input image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        class_name = classes[predicted.item()]
        image_url = images[class_name]
        description = descriptions[class_name]
        return jsonify({'prediction': class_name, 'image_url': image_url, 'description': description})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
