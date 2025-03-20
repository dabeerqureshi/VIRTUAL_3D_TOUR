from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import faiss
import torch
import clip
import uuid
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50

app = Flask(__name__, static_folder="static", template_folder="templates")

# Define folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
IMAGE_FOLDER = os.path.join(BASE_DIR, "static/3d_images")
DATABASE_FOLDER = os.path.join(BASE_DIR, "database")
VALID_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}  # Add more if needed


# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Load precomputed embeddings and image names
image_embeddings, image_names = None, None
try:
    image_embeddings = np.load(os.path.join(DATABASE_FOLDER, "image_embeddings.npy")).astype("float32")
    image_names = np.load(os.path.join(DATABASE_FOLDER, "image_names.npy"))
    print("🔹 Image embeddings and names loaded successfully!")
except Exception as e:
    print(f"❌ Error loading embeddings: {e}")

# Initialize FAISS Index
index = None
if image_embeddings is not None:
    d = image_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.add(image_embeddings)
    print("✅ FAISS index initialized!")

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet_model = resnet50(pretrained=True).to(device)
resnet_model.fc = torch.nn.Identity()
resnet_model.eval()
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def extract_image_embedding(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = resnet_model(image).cpu().numpy()
        return embedding.astype("float32")
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

def extract_text_embedding(text):
    try:
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(device)
            text_embedding = clip_model.encode_text(text_tokens)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        return text_embedding.cpu().numpy().astype("float32")
    except Exception as e:
        print(f"❌ Error processing text embedding: {e}")
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/gallery')
def gallery():
    return render_template('gallery.html')


@app.route("/get_3d_visual", methods=["POST"])
def get_3d_visual():
    place_name = request.form.get("place_name", "").strip()
    place_desc = request.form.get("place_desc", "").strip()
    file = request.files.get("place_image")

    print(f"🔹 Received Input - Name: {place_name}, Description: {place_desc}, Image: {file}")

    # Validate input: Ensure only ONE input is provided
    inputs = [bool(place_name), bool(place_desc), file is not None]
    if sum(inputs) != 1:
        print("❌ Error: Multiple or No inputs provided.")
        return jsonify({"error": "Please provide only one input at a time."}), 400

    # 🔹 Case 1: Search by Place Name
    if place_name:
        print(f"🔍 Searching for 3D video by name: {place_name}")
        
        matching_files = [
            video for video in os.listdir(IMAGE_FOLDER)
            if place_name.lower() in os.path.splitext(video)[0].lower()
            and os.path.splitext(video)[1].lower() in VALID_VIDEO_EXTENSIONS
        ]

        if matching_files:
            print(f"✅ Match found: {matching_files[0]}")
            return jsonify({"3D_video": f"/static/3d_images/{matching_files[0]}"}), 200
        else:
            print("❌ No matching 3D video found.")
            return jsonify({"error": f"No matching 3D video found for '{place_name}'"}), 404

    # 🔹 Case 2: Search by Image
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)

        # Extract embedding
        query_embedding = extract_image_embedding(file_path)
        if query_embedding is None:
            return jsonify({"error": "Failed to process image."}), 500
        
        query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

        # **Debugging Prints**
        print(f"🔹 FAISS index dimension: {index.d}")
        print(f"🔹 Query embedding shape before correction: {query_embedding.shape}")

        # **Ensure Query Embedding Matches FAISS Index**
        if query_embedding.shape[1] < index.d:
            # **Pad with zeros if smaller**
            query_embedding = np.pad(query_embedding, ((0, 0), (0, index.d - query_embedding.shape[1])), 'constant')
        elif query_embedding.shape[1] > index.d:
            # **Trim if larger**
            query_embedding = query_embedding[:, :index.d]

        # **Normalize Query Embedding**
        faiss.normalize_L2(query_embedding)

        print(f"🔹 Query embedding shape after correction: {query_embedding.shape}")

        # **Perform FAISS Search**
        _, I = index.search(query_embedding, 1)
        best_match_index = I[0][0]

        if 0 <= best_match_index < len(image_names):
            matched_class = image_names[best_match_index]
            video_path = os.path.join(IMAGE_FOLDER, f"{matched_class}.mp4")

            if os.path.exists(video_path):
                return jsonify({"3D_visual": video_path}), 200
            else:
                return jsonify({"error": "Matching 3D video not found."}), 404

        return jsonify({"error": "No matching 3D visual found."}), 404

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
    # 🔹 Case 3: Search by Text Description
    if place_desc:
        query_embedding = extract_text_embedding(place_desc)

        if query_embedding is None:
            return jsonify({"error": "Failed to process the text input."}), 500

        _, I = index.search(query_embedding, 1)
        best_match = I[0][0]

        if 0 <= best_match < len(image_names):
            return jsonify({"3D_visual": f"/static/3d_images/{image_names[best_match]}"}), 200

        return jsonify({"error": "No matching 3D visual found."}), 404

    return jsonify({"error": "Invalid input"}), 400

if __name__ == "__main__":
    app.run(debug=True)
