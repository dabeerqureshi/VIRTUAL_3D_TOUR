import numpy as np
import faiss

# Generate random embeddings for testing
num_images = 100
embedding_size = 128

image_embeddings = np.random.rand(num_images, embedding_size).astype("float32")
image_names = np.array([f"place_{i}.jpg" for i in range(num_images)])

np.save("backend/database/image_embeddings.npy", image_embeddings)
np.save("backend/database/image_names.npy", image_names)

# Create FAISS index
index = faiss.IndexFlatL2(embedding_size)
index.add(image_embeddings)

faiss.write_index(index, "backend/database/embeddings.index")

print("FAISS index created and saved.")
