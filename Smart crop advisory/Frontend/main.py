from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import random
import heapq   # ✅ Heap (Priority Queue)

app = FastAPI()

#  CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Crop Data (HashMap)
crop_data = {

"Rice": {"temp": "25-35°C", "humidity": "60-80%", "duration": "3-6 months",
         "fertilizer": [{"name": "Urea", "desc": "Nitrogen for growth"},
                        {"name": "DAP", "desc": "Root development"}]},

"Wheat": {"temp": "15-25°C", "humidity": "40-60%", "duration": "4-5 months",
          "fertilizer": [{"name": "NPK", "desc": "Balanced nutrients"}]},

"Maize": {"temp": "20-30°C", "humidity": "50-70%", "duration": "3 months",
          "fertilizer": [{"name": "Urea", "desc": "Boost growth"}]},

"Barley": {"temp": "12-25°C", "humidity": "40-60%", "duration": "3 months",
           "fertilizer": [{"name": "NPK", "desc": "Balanced fertilizer"}]},

"Millets": {"temp": "20-30°C", "humidity": "40-50%", "duration": "2-3 months",
            "fertilizer": [{"name": "Organic Compost", "desc": "Improves soil"}]},

"Chickpea": {"temp": "20-25°C", "humidity": "40-60%", "duration": "3 months",
             "fertilizer": [{"name": "Phosphorus", "desc": "Root growth"}]},

"Groundnut": {"temp": "25-30°C", "humidity": "50-60%", "duration": "3-4 months",
              "fertilizer": [{"name": "Gypsum", "desc": "Pod development"}]},

"Mustard": {"temp": "10-25°C", "humidity": "40-60%", "duration": "3 months",
            "fertilizer": [{"name": "Sulphur", "desc": "Oil formation"}]},

"Tomato": {"temp": "20-30°C", "humidity": "50-70%", "duration": "2-3 months",
           "fertilizer": [{"name": "Compost", "desc": "Organic nutrients"}]},

"Potato": {"temp": "15-20°C", "humidity": "60-70%", "duration": "3 months",
           "fertilizer": [{"name": "Potassium", "desc": "Tuber quality"}]},

"Onion": {"temp": "15-25°C", "humidity": "50-70%", "duration": "3-4 months",
          "fertilizer": [{"name": "Nitrogen", "desc": "Bulb growth"}]},

"Mango": {"temp": "24-30°C", "humidity": "50-60%", "duration": "3-5 years",
          "fertilizer": [{"name": "FYM", "desc": "Soil enrichment"}]},

"Banana": {"temp": "25-35°C", "humidity": "60-80%", "duration": "9-12 months",
           "fertilizer": [{"name": "Potassium", "desc": "Fruit size"}]},

"Guava": {"temp": "20-30°C", "humidity": "50-70%", "duration": "2-3 years",
          "fertilizer": [{"name": "NPK", "desc": "Balanced nutrients"}]},

"Coconut": {"temp": "25-30°C", "humidity": "70-80%", "duration": "5-7 years",
            "fertilizer": [{"name": "Organic", "desc": "Soil health"}]},

"Cotton": {"temp": "25-35°C", "humidity": "50-60%", "duration": "5-6 months",
           "fertilizer": [{"name": "Potassium", "desc": "Fiber quality"}]},

"Sugarcane": {"temp": "20-35°C", "humidity": "60-80%", "duration": "10-12 months",
              "fertilizer": [{"name": "Urea", "desc": "Fast growth"}]}
}

#  Graph (Soil → Crops)
graph = {
    "clay": ["Rice", "Sugarcane"],
    "sandy": ["Groundnut", "Millets"],
    "loamy": ["Wheat", "Maize", "Tomato"],
    "black": ["Cotton", "Sugarcane"]
}

#  Cache (HashMap)
cache = {}

#  Weather API
def get_weather(lat, lon):
    API_KEY = "e70fd66730c25382f8889fb0a4839688"

    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"

    response = requests.get(url)
    data = response.json()

    if "main" not in data:
        return 25, 60

    return data["main"]["temp"], data["main"]["humidity"]


# DFS (Graph Traversal)
def get_options(soil):
    visited = set()
    result = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)

        for n in graph.get(node, []):
            result.append(n)
            dfs(n)

    dfs(soil)
    return result


#  Scoring function
def score_crop(crop, temp, humidity):
    score = 0

    if "25-35" in crop_data[crop]["temp"]:
        score += 2
    if "60-80" in crop_data[crop]["humidity"]:
        score += 2

    return score


#  Home
@app.get("/")
def home():
    return {"message": "Backend running"}


#  Prediction
@app.post("/predict")
def predict(data: dict):

    soil = data.get("soil")
    irrigation = data.get("irrigation")
    investment = int(data.get("amount") or 0)
    lat = data.get("lat")
    lon = data.get("lon")

    # ⚡ Cache check
    key = (soil, lat, lon)
    if key in cache:
        return cache[key]

    #  Weather
    temp, humidity = get_weather(lat, lon)

    # Graph + DFS
    options = get_options(soil)

    if not options:
        options = list(crop_data.keys())

    #  Heap (Priority Queue)
    heap = []
    for c in options:
        score = score_crop(c, temp, humidity)
        heapq.heappush(heap, (-score, c))

    # Top-K
    top_k = 3
    top_crops = []

    for _ in range(min(top_k, len(heap))):
        top_crops.append(heapq.heappop(heap)[1])

    crop = top_crops[0]

    # Advice
    if temp > 35:
        advice = "High temperature — increase irrigation"
    elif humidity < 40:
        advice = "Low humidity — ensure watering"
    else:
        advice = "Conditions are favorable"

    result = {
        "crop": crop,
        "top_crops": top_crops,  
        "duration": crop_data.get(crop, {}).get("duration"),
        "current_temp": temp,
        "current_humidity": humidity,
        "recommended_temp": crop_data.get(crop, {}).get("temp"),
        "recommended_humidity": crop_data.get(crop, {}).get("humidity"),
        "fertilizers": crop_data.get(crop, {}).get("fertilizer"),
        "advice": advice
    }

    # ⚡ Store cache
    cache[key] = result

    return result