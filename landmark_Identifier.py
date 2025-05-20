import os
import tempfile
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import requests
from io import BytesIO
import logging

# Load environment variables from a .env file if needed
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LandmarkClassifier:
    """
    A class to download a Keras model from GCS, load it locally, and use it to predict
    whether an image is 'exterior' or 'interior'.
    """
    
    def __init__(self, labels=None):
        """
        Initializes the LandmarkClassifier.
        
        Args:
            labels (list, optional): A list of labels to map the output.
                                     Defaults to ["exterior", "interior"].
        """
        # Set the GCS bucket and model blob path
        self.bucket_name = 'sophiq_static_files'
        self.model_blob_name = 'landmark_classifier.h5'
        # Use provided labels or default
        self.labels = labels if labels is not None else ["exterior", "interior"]
        # Download and load the model locally from GCS
        self.model = self._load_model_via_local_download()

    def _download_model_from_gcs(self, local_path):
        """
        Downloads the model file from GCS to a local file.
        
        Args:
            local_path (str): Local file path where the model will be saved.
        """
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(self.model_blob_name)
        blob.download_to_filename(local_path)
        logger.debug(f"Model downloaded to {local_path}")

    def _load_model_via_local_download(self):
        """
        Downloads the model from GCS to a temporary file and loads it.
        
        Returns:
            tf.keras.Model: The loaded Keras model.
        """
        # Create a temporary file to hold the model
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
            temp_model_path = temp_file.name

        # Download the model from GCS to the temporary file
        self._download_model_from_gcs(temp_model_path)
        # Load the model using TensorFlow
        model = tf.keras.models.load_model(temp_model_path)
        # Remove the temporary file after loading the model
        os.remove(temp_model_path)
        return model

    def _load_image(self, image_input):
        """
        Loads an image from a file path or URL.
        
        Args:
            image_input: Path to image file, URL, or PIL Image object
            
        Returns:
            PIL.Image: Loaded image in RGB format
        """
        if isinstance(image_input, str):
            if image_input.startswith("http://") or image_input.startswith("https://"):
                response = requests.get(image_input)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input. Provide path, URL, or PIL Image.")
        return image

    def preprocess_image(self, image, target_size=(224, 224)):
        """
        Preprocesses an image for prediction.
        
        Args:
            image: PIL Image object
            target_size (tuple): The target size to which the image is resized.
        
        Returns:
            np.ndarray: Preprocessed image array ready for model prediction.
        """
        # Resize the image to the target size
        img = image.resize(target_size)
        # Convert the image to a numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        # Expand dimensions to create a batch of size 1
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, image_path):
        """
        Preprocesses an image and uses the loaded model to predict the class label.
        
        Args:
            image_path (str): The local path to the image file.
        
        Returns:
            str: The predicted label (e.g., "exterior" or "interior").
        """
        img = self._load_image(image_path)
        processed_image = self.preprocess_image(img)
        
        # Suppress TensorFlow output during prediction
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        prediction = self.model.predict(processed_image, verbose=0)
        
        # Debug logging instead of printing
        logger.debug(f"Raw model output: {prediction}")
        
        # If the model output is a single probability (sigmoid output)
        if prediction.shape[-1] == 1:
            prob = prediction[0][0]
            logger.debug(f"Sigmoid probability: {prob:.4f}")
            label = self.labels[1] if prob > 0.5 else self.labels[0]
            confidence = prob if prob > 0.5 else 1 - prob
        else:
            # For softmax outputs, pick the class with highest probability
            class_idx = np.argmax(prediction, axis=-1)[0]
            confidence = prediction[0][class_idx]
            logger.debug(f"Softmax prediction index: {class_idx}, confidence: {confidence:.4f}")
            label = self.labels[class_idx]
        return label, float(confidence)
    
    def predict_from_image(self, image_path):
        """
        Preprocesses a single image, applies the model to predict the class,
        and displays the image with the predicted label.
        
        Args:
            image_path (str): The local path to the image file.
        """
        # Get the predicted label
        label, confidence = self.predict(image_path)
        
        # Load the image and display it with matplotlib
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Predicted: {label} (Confidence: {confidence:.2f})")
        plt.show()
    
    def classify_image(self, image=False, url=False):
        """
        Classify an image as 'exterior' or 'interior'.
        
        Args:
            image: PIL Image or path to local image file (set to False if using URL)
            url: URL of the image (set to False if using local image)
            
        Returns:
            tuple: (predicted_class, confidence)
        """
        if image and not url:
            return self.predict(image)
        elif url and not image:
            return self.predict(url)
        else:
            raise ValueError("Provide either 'image' or 'url', but not both or neither")

# ----------------- Example Usage -----------------
"""
from LandMark_classifier.landmark_Identifier import LandmarkClassifier

# Initialize the classifier
classifier = LandmarkClassifier()

# Example with a local image path
result_local, confidence_local = classifier.classify_image(image="path/to/local/image.jpg")
print(f"Prediction: {result_local}, Confidence: {confidence_local:.2f}")

# Example with a URL
result_url, confidence_url = classifier.classify_image(url="https://example.com/image.jpg")
print(f"Prediction: {result_url}, Confidence: {confidence_url:.2f}")

# Example with a PIL Image
from PIL import Image
img = Image.open("path/to/image.jpg")
result_pil, confidence_pil = classifier.classify_image(image=img)
print(f"Prediction: {result_pil}, Confidence: {confidence_pil:.2f}")
"""