# Import necessary Django modules
from django.contrib.auth import authenticate, login
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2
from django.shortcuts import render
from django.contrib.auth.models import User
from threading import Thread
from urllib import request
import flwr as fl
from tensorflow.keras.utils import to_categorical
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .forms import SignupForm

# Import necessary libraries for image processing and machine learning
from neo4j import GraphDatabase
import numpy as np
import cv2
import json
import tensorflow as tf
from .models import KerasModel
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from django.http import JsonResponse, HttpResponse
from datetime import datetime, timezone
import threading
import os
from dp import train_with_dp
from fhe import encrypt_embedding_for_neo4j
from my_flower_client import FlowerClient
import shutil


# Import Django views and decorators
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

# Import machine learning metrics
from sklearn.metrics.pairwise import cosine_similarity
from keras_facenet import FaceNet

# Define URL patterns for the application
from django.urls import path

NEO4J_URI = 'neo4j+s://abf9edae.databases.neo4j.io'
NEO4J_USER = 'neo4j'
NEO4J_PASSWORD = '7LbESA2foTba6tOIh5I4V2R9l_65t2PeI1IG1Vq2-8I'


def home_page(request):
    return render(request, 'test1.html')

# Singleton for Model Management


class ModelSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelSingleton, cls).__new__(cls)
                    cls._instance.loaded_model = cls._load_latest_model()
        return cls._instance

    @staticmethod
    def _load_latest_model():
        """Load the latest model from the database."""
        keras_model_instance = KerasModel.objects.last()
        if keras_model_instance:
            try:
                model_path = keras_model_instance.model_file.path
                print(f"Loading model from {model_path}.")
                return load_model(model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("No Keras model found in the database.")
        return None


# Directory for Local Cache
LOCAL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".keras_model_cache")
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)


def get_cached_model():
    """Download and cache the model locally, or load from cache."""
    model_file_name = "my_model.keras"
    model_path = os.path.join(LOCAL_CACHE_DIR, model_file_name)

    if os.path.exists(model_path):
        print("Loading model from cache...")
        return load_model(model_path)

    print("Downloading model...")
    keras_model_instance = KerasModel.objects.last()
    if keras_model_instance:
        shutil.copy(keras_model_instance.model_file.path, model_path)
        print("Model cached successfully.")
        return load_model(model_path)

    print("No model found in database.")
    return None


def load_model():
    """Load MobileNetV2 model with a bottleneck layer for feature extraction."""
    base_model = MobileNetV2(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add Global Average Pooling layer followed by dense layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(640, activation='relu')(x)  # First reduction step
    x = Dense(480, activation='relu')(x)  # Second reduction step

    # Final bottleneck layer to reduce to 384 dimensions
    bottleneck_layer = Dense(384, activation='relu')(x)

    model = Model(inputs=base_model.input,
                  outputs=bottleneck_layer)  # Create final model
    return model


def get_cached_model():
    """Placeholder function for loading a cached model."""
    # Implement your caching logic here. For now, we will just call load_model.
    return load_model()


def get_model(mode="feature_extraction"):
    """Load the model for feature extraction or training."""
    model = get_cached_model()
    print("Cached model loaded")

    if not model:
        return None

    if mode == "feature_extraction":
        # Create a new model that outputs the bottleneck layer for feature extraction
        feature_extraction_model = tf.keras.Model(
            inputs=model.input,
            # Output from the last dense layer (bottleneck)
            outputs=model.layers[-1].output
        )
        # Get the output shape of the feature vector
        output_shape = feature_extraction_model.output_shape
        print("Feature vector dimensionality:", output_shape)
        return feature_extraction_model

    elif mode == "training":
        return model
    else:
        raise ValueError(
            "Invalid mode. Choose 'feature_extraction' or 'training'."
        )


def process_image(image_data):
    """Preprocess image for MobileNetV2."""
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (224, 224))
    img_preprocessed = preprocess_input(img_resized.astype('float32'))
    return np.expand_dims(img_preprocessed, axis=0)


def extract_features(model, image_data):
    """Extract features from the global pooling layer."""
    img_array = process_image(image_data)  # Process the image
    # Load the feature extraction model
    feature_model = get_model(mode="feature_extraction")
    print("Model loaded in extract_features function")
    if feature_model:
        # Predict features using the model
        features = feature_model.predict(img_array)
        print("Extracted features:", features)
        print("Feature vector dimensionality:",
              features.shape)  # Shape of the output
        # Total number of elements in the array
        print("Size of feature vector:", features.size)
        return features  # Return the extracted features

    return None


def start_flower_client(image_data, username):
    """Start a Flower client for federated learning."""

    # Convert image_data to numpy array if necessary (assuming it's already in the correct format)
    img_array = np.array(image_data)  # Ensure it's a NumPy array
    print(f"The shape is {img_array.shape}")  # Check shape
    print(f"The data is {img_array}")  # Check contents
    print(f"The datatype is {img_array.dtype}")  # Check contents

    # Train with differential privacy using dp.py
    print("Starting differential privacy training...")

    updated_model_params = train_with_dp(
        img_array)  # Call the DP training function

    # Create a Flower client with the trained model parameters
    # Pass only the model parameters
    client = FlowerClient(model_params=updated_model_params)

    print("Federated learning update sent to server.")

    return client


class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_user_node(self, username):
        with self.driver.session() as session:
            session.run("CREATE (p:Person {id: $username})", username=username)

    def create_image_node(self, username, embedding):
        creating_at = datetime.now(timezone.utc).strftime(
            "Date:%d-%m-%Y UTC:%H:%M:%S")

        with self.driver.session() as session:
            # Flatten the embedding and convert to a list
            flat_embedding = embedding
            session.run(
                """
                MATCH (p:Person {id: $username})
                CREATE (img:Image {id: $username, embedding: $embedding, created_at: $created_at})
                CREATE (p)-[:HAS_IMAGE]->(img)
                """,
                username=username,
                embedding=flat_embedding,  # Store as list directly
                created_at=creating_at
            )

    def get_user_embeddings(self, username):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Person {id: $username})-[:HAS_IMAGE]->(img:Image)
                RETURN img.embedding AS embedding, img.created_at AS created_at
                """,
                username=username
            )
            return [
                {
                    # Decode JSON string back to list
                    "embedding": np.array(record["embedding"]),
                    "created_at": record["created_at"]
                }
                for record in result
            ]

    def find_similar_person(self, new_embedding):
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Person)-[:HAS_IMAGE]->(img:Image) RETURN p.id AS person_id, img.embedding AS embedding")
            embeddings = [(record["person_id"], np.array(
                record["embedding"])) for record in result]

        # Calculate cosine similarities
        similarities = [(person_id, cosine_similarity([new_embedding], [embedding])[0][0])
                        for person_id, embedding in embeddings]
        # Sort and return the top 3
        similar_persons = sorted(
            similarities, key=lambda x: x[1], reverse=True)[:10]
        return similar_persons
#####################################


def signup_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        first_name = request.POST.get('first_name')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Validation check
        if User.objects.filter(username=username, first_name=first_name).exists():
            return HttpResponse('The combination of username and first name must be unique.')

        # Create user
        user = User(username=username, first_name=first_name, email=email)
        user.set_password(password)
        user.save()

        # Neo4j integration
        neo4j_handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        try:
            neo4j_handler.create_user_node(user.username)
        finally:
            neo4j_handler.close()

        return redirect('login')

    return render(request, 'signup.html')


def loginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('landing')
        else:
            return render(request, 'login.html', {
                'error_message': 'Invalid username or password'
            })

    return render(request, 'login.html')


def background_model_download():
    """
    Background task to download and cache the model.
    """
    try:
        model = get_model()  # This will download and cache the model
        if model:
            print("Model downloaded and cached successfully in the background.")
        else:
            print("Failed to load the model in the background.")
    except Exception as e:
        print(f"Error during background model download: {e}")


def logoutPage(request):
    logout(request)
    return redirect('login')


@login_required
def landingPage(request):
    neo4j_handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Retrieve existing embeddings for the logged-in user
    try:
        embeddings_data = neo4j_handler.get_user_embeddings(
            request.user.username)
        num_embeddings = len(embeddings_data)
        embedding_info = [{"embedding_id": idx + 1, "created_at": record["created_at"]}
                          for idx, record in enumerate(embeddings_data)]
    finally:
        neo4j_handler.close()

    context = {
        "username": request.user.username,
        "num_embeddings": num_embeddings,
        "embedding_info": embedding_info,
        "max_embeddings": 5,
    }

    if request.method == 'POST':
        try:
            # Handle uploaded image
            image_file = request.FILES.get('image')
            if not image_file:
                return HttpResponse("No image file received.", status=400)

            image_bytes = image_file.read()

            # Load the locally cached model
            model = get_model(mode="feature_extraction")
            if not model:
                return JsonResponse({"error": "Model could not be loaded."}, status=500)

            # Process the image to get embedding
            embedding = extract_features(model, image_bytes)
            if embedding is None:
                return JsonResponse({"error": "Failed to generate embedding."}, status=500)

            encrypted_embedding = encrypt_embedding_for_neo4j(embedding)
            if encrypted_embedding is None:
                return JsonResponse({"error": "Failed to encrypt embedding."}, status=500)

            # Save embedding to Neo4j
            neo4j_handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            try:
                neo4j_handler.create_image_node(
                    request.user.username, encrypted_embedding)
            finally:
                neo4j_handler.close()
            print("saved successfully!")

            # Start the federated learning client and send updated model to server
            # This connects to the federated server
            client = start_flower_client(
                image_bytes, request.user.username)
            print("Federated learning client initialized and model update sent.")

            return JsonResponse({"message": "Image processed, embedding stored, and local training completed."})

        except Exception as e:
            return JsonResponse({"error": f"Error processing image: {e}"}, status=500)

    # Handle user logout to update the central model
    if "logout" in request.POST:
        try:
            # Load the locally updated model
            model = get_cached_model()
            if model:
                # Save the updated model to the database
                keras_model_instance = KerasModel.objects.last()
                if keras_model_instance:
                    model.save(keras_model_instance.model_file.path)
                    print("Central model updated successfully.")
                return JsonResponse({"message": "Central model successfully updated."})
        except Exception as e:
            return JsonResponse({"error": f"Error updating central model: {e}"}, status=500)

    return render(request, 'landing.html', context)


@login_required
def delete_embedding(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # This is for frontend display purposes
            embedding_id = data.get('embedding_id')
            # This is used for identifying the embedding to delete
            created_at = data.get('created_at')
            if not created_at:
                return JsonResponse({"error": "created_at is required."}, status=400)
            # Connect to Neo4j and delete the embedding based on the created_at timestamp
            neo4j_handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            try:
                with neo4j_handler.driver.session() as session:
                    result = session.run(
                        """
                        MATCH (p:Person {id: $username})-[:HAS_IMAGE]->(img:Image {created_at: $created_at})
                        DETACH DELETE img
                        RETURN COUNT(img) AS deletedCount
                        """,
                        username=request.user.username,
                        created_at=created_at
                    )
                    deleted_count = result.single()["deletedCount"]
                    if deleted_count == 0:
                        return JsonResponse({"error": "No matching embedding found to delete."}, status=404)
            finally:
                neo4j_handler.close()
            return JsonResponse({"message": "Embedding deleted successfully."})
        except Exception as e:
            return JsonResponse({"error": f"Error deleting embedding: {e}"}, status=500)
    return JsonResponse({"error": "Invalid request method."}, status=400)

# Added by Samuel


class FaceRecognitionAPI(View):
    camera_thread = None
    stop_event = threading.Event()
    recognized_persons = []

    @method_decorator(csrf_exempt)
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)

    @classmethod
    def start_camera(cls):
        cls.stop_event.clear()
        cap = cv2.VideoCapture(0)  # Start camera
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        neo4j_handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        # Load the feature extraction model
        feature_extraction_model = get_model(mode="feature_extraction")

        while not cls.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                continue

            display_text = "No face detected"  # Default message

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                face = np.expand_dims(face, axis=0)

                new_embedding = feature_extraction_model.predict(face)[0]

                similar_persons = neo4j_handler.find_similar_person(
                    new_embedding)
                cls.recognized_persons = [{"id": person_id, "similarity": similarity}
                                          for person_id, similarity in similar_persons]

                # Take the first recognized person (highest similarity)
                if cls.recognized_persons:
                    # Get the top 3 recognized persons
                    # Get top 3 matches
                    top_matches = cls.recognized_persons[:3]
                    display_text = "\n".join([f"ID: {match['id']}, Similarity: {
                                             match['similarity']:.2f}" for match in top_matches])
                else:
                    display_text = "Unknown person"

                # Overlay text on the OpenCV frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (0, 0, 255)  # Red color in BGR format
                thickness = 1
                position = (10, 30)  # Top-left corner

                # Split display_text into lines for better positioning
                for i, line in enumerate(display_text.split('\n')):
                    cv2.putText(frame, line, (position[0], position[1] + i * 20), font,
                                font_scale, font_color, thickness, cv2.LINE_AA)

            # Display the OpenCV frame
            cv2.imshow("Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        neo4j_handler.close()

    @classmethod
    def stop_camera(cls):
        cls.stop_event.set()
        if cls.camera_thread and cls.camera_thread.is_alive():
            cls.camera_thread.join()
            cls.camera_thread = None

    def post(self, request, action):
        if action == "start":
            if self.camera_thread and self.camera_thread.is_alive():
                return JsonResponse({"error": "Recognition is already running."}, status=400)

            self.camera_thread = threading.Thread(target=self.start_camera)
            self.camera_thread.start()
            return JsonResponse({"message": "Recognition started."})

        elif action == "stop":
            self.stop_camera()
            return JsonResponse({
                "message": "Recognition stopped.",
                "recognized_persons": self.recognized_persons
            })

        return JsonResponse({"error": "Invalid action."}, status=400)
