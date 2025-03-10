from typing import Optional, Union
from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from captcha.image import ImageCaptcha
import io
import torch
from PIL import Image
from src.model import CaptchaModelCNN, CaptchaModelCRNN, CaptchaModelViT
from src.utils.utils import batchDecodeCTCOutput, batchOnehotDecodeLabel
from torchvision.transforms import v2
import psycopg2
import base64
import os


transforms = v2.Compose([
    v2.ToTensor(),
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
])

class Predictor():
    def __init__(self):
        self.model = None
        self.model_name = None
    def loadModel(self, model_name: str):
        # skip model loading if alread loaded
        if model_name == self.model_name:
            return
        # load the model
        if model_name == "cnn":
            self.model = CaptchaModelCNN()
            self.model.load_state_dict(torch.load(f"checkpoints/train_{model_name}/checkpoint.pth", map_location=torch.device('cpu')))
        elif model_name == "crnn":
            self.model = CaptchaModelCRNN()
            self.model.load_state_dict(torch.load(f"checkpoints/train_{model_name}/checkpoint.pth", map_location=torch.device('cpu'))['model_state_dict'])
        elif model_name == "vit":
            self.model = CaptchaModelViT()
            self.model.load_state_dict(torch.load(f"checkpoints/train_{model_name}/checkpoint.pth", map_location=torch.device('cpu'))['model_state_dict'])
        else:
            raise ValueError("Invalid model name")
        self.model_name = model_name
        self.model.eval()
    def predict(self, image: torch.Tensor) -> str:
        assert self.model_name is not None, "Model not loaded"
        # predict the text
        text = None
        with torch.no_grad():
            output = self.model(image.unsqueeze(0))
            if self.model_name == "vit" or self.model_name == "crnn":
                # decode the output using CTC decoder
                text = batchDecodeCTCOutput(output)
            else:
                # decode the output using argmax
                text = batchOnehotDecodeLabel(output.reshape(-1, 5, 36))
        return text[0]

predictor = Predictor()

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    return psycopg2.connect(
        dbname="captcha_db",
        user="postgres",
        password="password",
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432")
    )

def create_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS captcha_results (
            id SERIAL PRIMARY KEY,
            label TEXT,
            prediction TEXT,
            model TEXT,
            image BYTEA
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

create_table()

@app.get("/")
def home() -> Union[str, dict]:
    return {"message": "Hello World"}

# route to get a random captcha image
@app.get("/generate")
def get_captcha(text: str) -> Response:
    '''
    Generate a captcha image given a text of length 5
    '''
    if len(text) != 5:
        return Response(content="Text length should be 5", status_code=400, media_type="text/plain")
    # generate a random captcha image
    image_data = generate_captcha(text.strip().upper())
    return Response(content=image_data, media_type="image/png")
def generate_captcha(text, text_length=5):
    image = ImageCaptcha(width=160, height=60)
    image_data = image.generate_image(text)
    image_bytes = io.BytesIO()
    image_data.save(image_bytes, format="PNG")
    return image_bytes.getvalue()

# route to solve the captcha text
@app.post("/solve")
def solve_captcha(file: UploadFile, model_name: str, label: str = '') -> dict:
    '''
    Solve the captcha using the model and save the result
    '''
    # read the uploaded image
    image = Image.open(io.BytesIO(file.file.read()))
    # preprocess the image
    image = preprocess_image(image)
    # predict the text
    predictor.loadModel(model_name)
    text = predictor.predict(image)

    # Convert image to bytes
    image_bytes = io.BytesIO()
    file.file.seek(0)
    image_bytes.write(file.file.read())
    image_data = image_bytes.getvalue()

    # Store in PostgreSQL
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO captcha_results (label, prediction, model, image) VALUES (%s, %s, %s, %s)",
        (label.upper(), text, model_name, psycopg2.Binary(image_data))
    )
    conn.commit()
    cur.close()
    conn.close()

    return {"decoded": text}

def preprocess_image(image: Image.Image) -> torch.Tensor:
    '''
    Preprocess the image for the model
    '''
    # read the uploaded image
    image = transforms(image)
    return image

# route to fetch solved captchas
@app.get("/history")
def fetch_captchas() -> dict:
    '''
    Fetch all the solved captchas
    '''
    # Fetch the results from the database
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, label, prediction, model, image FROM captcha_results ORDER BY id DESC LIMIT 30")
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    # Format results
    formatted_results = []
    for result in results:
        image_base64 = base64.b64encode(result[4]).decode("utf-8")  # Convert binary image data to base64
        formatted_results.append({
            "id": result[0],
            "label": result[1],
            "prediction": result[2],
            "model": result[3],
            "image": f"data:image/png;base64,{image_base64}"  # Base64 encoded image with MIME type
        })

    return {"results": formatted_results}
