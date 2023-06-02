from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import numpy as np

# Schema for api
class PredictionRequest(BaseModel):
    cap_diameter: float
    cap_shape: int
    cap_surface: int
    cap_color: int
    does_bruise_or_bleed: int
    gill_attachment: int
    gill_spacing: int
    gill_color: int
    stem_height: float
    stem_width: float
    stem_root: int
    stem_surface: int
    stem_color: int
    veil_color: int
    has_ring: int
    ring_type: int
    spore_print_color: int
    habitat: int

# Model location
model = joblib.load('./RandomForestModel.pkl')

# FastAPI instance
app = FastAPI()

# Index route
@app.get("/")
def index():
    return {"message": "Use the /predict route to use the model"}

# /predict route with response
@app.post("/predict")
def predict_mushroom(poisonous: PredictionRequest):
    features = np.array([[poisonous.cap_diameter, poisonous.cap_shape, poisonous.cap_surface, poisonous.cap_color,
                          poisonous.does_bruise_or_bleed, poisonous.gill_attachment, poisonous.gill_spacing,
                          poisonous.gill_color, poisonous.stem_height, poisonous.stem_width, poisonous.stem_root,
                          poisonous.stem_surface, poisonous.stem_color, poisonous.veil_color, poisonous.has_ring,
                          poisonous.ring_type, poisonous.spore_print_color, poisonous.habitat]])
    prediction = model.predict(features)
    label = "poisonous" if prediction[0] == 1 else "edible"
    return {"prediction": int(prediction[0]), "label": label}

# For starting uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# For local testing: uvicorn api:app --reload
