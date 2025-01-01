# Import necessary Django modules
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

embedder = FaceNet()


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


def get_model(mode="feature_extraction"):
    """Load the model for feature extraction or training."""
    model = get_cached_model()
    print("Cached model loaded")
    if not model:
        return None

    if mode == "feature_extraction":
        feature_extraction_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer("global_average_pooling2d").output
        )
        # Get the output shape of the feature vector
        output_shape = feature_extraction_model.output_shape
        print("Feature vector dimensionality:", output_shape)
        return feature_extraction_model

    elif mode == "training":
        return model
    else:
        raise ValueError(
            "Invalid mode. Choose 'feature_extraction' or 'training'.")


def process_image(image_data):
    """Preprocess image for MobileNetV2."""
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (224, 224))
    img_preprocessed = preprocess_input(img_resized.astype('float32'))
    return np.expand_dims(img_preprocessed, axis=0)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, train_labels):
        self.model = model
        self.train_data = train_data
        self.train_labels = train_labels

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.train_data, self.train_labels,
                       epochs=1, batch_size=32)
        return self.get_parameters(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(
            self.train_data, self.train_labels)
        return loss, len(self.train_data), {"accuracy": accuracy}


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
    # Get the model for training
    model = get_model(mode="training")
    img_array = process_image(image_data)

    # Print the shape of the processed image
    # Should be (1, 224, 224, 3)
    print("Processed image shape:", img_array.shape)

    # Create a dummy label (not used for classification)
    s = model.output.shape
    dummy_labels = np.ones((img_array.shape[0], s[1]))
    print("Dummy labels shape:", dummy_labels.shape)

    # Create a Flower client with the model and data
    client = FlowerClient(model, img_array, dummy_labels)

    # Push the updated model to the federated server
    client.fit(client.get_parameters(), config={})

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
            flat_embedding = embedding.flatten().tolist()  # Ensure it's a flat list

            # Optionally convert to JSON string for storage
            embedding_json = json.dumps(
                flat_embedding)  # Convert to JSON string

            session.run(
                """
                MATCH (p:Person {id: $username})
                CREATE (img:Image {id: $username, embedding: $embedding, created_at: $created_at})
                CREATE (p)-[:HAS_IMAGE]->(img)
                """,
                username=username,
                embedding=embedding_json,  # Use JSON string representation
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
                    "embedding": json.loads(record["embedding"]),
                    "created_at": record["created_at"]
                }
                for record in result
            ]

    ################### Added by Samuel##################
    # def find_similar_person(self, new_embedding):
    #     def normalize(embedding):
    #         return embedding / np.linalg.norm(embedding)

    #     new_embedding = normalize(new_embedding)
    #     with self.driver.session() as session:
    #         result = session.run("MATCH (p:Person)-[:HAS_IMAGE]->(img:Image) RETURN p.id AS person_id, img.embedding AS embedding")
    #         embeddings = [(record["person_id"], normalize(np.array(record["embedding"]))) for record in result]

    #     similarities = [(person_id, cosine_similarity([new_embedding], [embedding])[0][0]) for person_id, embedding in embeddings]
    #     similar_persons = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    #     return similar_persons

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
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            # Hash the password
            user.set_password(form.cleaned_data['password'])
            user.save()
            # Create a Neo4j node for the new user
            neo4j_handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            try:
                neo4j_handler.create_user_node(user.username)
            finally:
                neo4j_handler.close()
            # Redirect to a success page or login page
            return HttpResponse("Successfully Signed Up!") and redirect('login')
    else:
        form = SignupForm()

    return render(request, 'signup.html', {'form': form})


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


def loginPage(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)

                # Start the background thread for model caching
                Thread(target=background_model_download).start()

                return redirect('landing')  # Redirect to your landing page
            else:
                return HttpResponse("Username or Password is incorrect!")
    else:
        form = AuthenticationForm()

    return render(request, 'login.html', {'form': form})


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

            # Save embedding to Neo4j
            neo4j_handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            try:
                neo4j_handler.create_image_node(
                    request.user.username, embedding)
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

        while not cls.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                continue

            display_text = "No face detected"  # Default message

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))
                face = np.expand_dims(face, axis=0)

                new_embedding = embedder.embeddings(face)[0]

                similar_persons = neo4j_handler.find_similar_person(
                    new_embedding)
                cls.recognized_persons = [{"id": person_id, "similarity": similarity}
                                          for person_id, similarity in similar_persons]

                # Take the first recognized person (highest similarity)
                if cls.recognized_persons:
                    top_match = cls.recognized_persons[0]
                    person_id = top_match['id']
                    similarity = top_match['similarity']
                    display_text = f"ID: {
                        person_id}, Similarity: {similarity:.2f}"
                else:
                    display_text = "Unknown person"

                break  # Process one face at a time

            # Overlay text on the OpenCV frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (0, 255, 0)  # Green
            thickness = 1
            position = (10, 30)  # Top-left corner

            cv2.putText(frame, display_text, position, font,
                        font_scale, font_color, thickness, cv2.LINE_AA)

            # Display the OpenCV frame
            cv2.imshow("Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        neo4j_handler.close()

    # def start_camera(cls):
    #     cls.stop_event.clear()
    #     cap = cv2.VideoCapture(0)  # Start camera
    #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #     neo4j_handler = Neo4jHandler(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    #     while not cls.stop_event.is_set():
    #         ret, frame = cap.read()
    #         if not ret:
    #             continue

    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #         if len(faces) == 0:
    #             continue

    #         for (x, y, w, h) in faces:
    #             face = frame[y:y+h, x:x+w]
    #             face = cv2.resize(face, (160, 160))
    #             face = np.expand_dims(face, axis=0)
    #             new_embedding = embedder.embeddings(face)[0]

    #             similar_persons = neo4j_handler.find_similar_person(new_embedding)
    #             cls.recognized_persons = [{"id": person_id, "similarity": similarity}
    #                                       for person_id, similarity in similar_persons]
    #             break  # Process one face at a time

    #         # Display OpenCV frame
    #         cv2.imshow("Recognition", frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    #     cap.release()
    #     cv2.destroyAllWindows()
    #     neo4j_handler.close()

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
