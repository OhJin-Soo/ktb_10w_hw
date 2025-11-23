from fastapi import FastAPI, Form
from PIL import Image, ImageFile
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import base64

app = FastAPI()
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1️⃣ 모델 로드
device = torch.device("cpu")
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(512, 3)  # 클래스 수 3개 예시
model.to(device)
model.eval()

# 클래스 이름 예시
class_names = ["good", "bad", "risk"]

# 2️⃣ 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 3️⃣ 추론 API
@app.post("/predict")
async def predict(image_base64: str = Form(...)):
    try:
        # base64 → PIL 이미지 변환
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
    except Exception:
        return {"error": "Invalid image input"}

    # 전처리 및 추론
    input_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_class_idx]
        risk = torch.sum(probabilities * torch.tensor([1.0, 0.0, 0.5])).item()

    # 결과 반환
    return {
        "상태": predicted_class,
        "확률": probabilities.tolist(),
        "위험도": risk
    }

# 4️⃣ 로컬 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
