 
       Index.html: 
        <!DOCTYPE html> 
        <html lang="en"> 
 <head> 
             <meta charset="UTF-8"> 
             <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
        <title>Paddy Disease Detection</title> 
             <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}"> 
       </head> 
       <body> 
     <!-- Navigation Bar --> 
     <nav class="navbar"> 
          <div class="nav-left"> 
              <h1 class="nav-title">Paddy Disease Detection</h1> 
          </div> 
         <div class="nav-right"> 
             <button class="nav-button">Home</button> 
             <button class="nav-button">About Paddy</button> 
             <button class="nav-button">Sign In</button> 
             <button class="nav-button">Sign Up</button> 
             <button class="nav-button profile-button">Profile</button> 
         </div> 
     </nav> 
     <!-- Main Content --> 
     <div class="main-content"> 
7 | P a g e  
          <h1>Paddy Disease Detection</h1> 
          <form action="/predict" method="post" enctype="multipart/form-data"> 
             <label for="file-upload" class="upload-label">Choose File</label> 
             <input id="file-upload" type="file" name="file" accept="image/*" required> 
             <button type="submit">Upload Image</button> 
         </form> 
        {% if disease %} 
            <div id="result-section"> 
                <h2>Detected Disease: {{ disease }}</h2> 
                <img src="{{ image_path }}" alt="Uploaded Image" id="uploaded-image"> 
                <div class="buttons"> 
                    <button onclick="showMedicine()">Medicine Recommendation</button> 
                    <button onclick="showSteps()">Steps to Reduce Disease</button> 
                    <button onclick="deleteImage()">Delete Image</button> 
                </div> 
                <div id="medicine"> 
                    <h3>Medicine Recommendation:</h3> 
                    <p>{{ medicine }}</p> 
                </div> 
                <div id="steps"> 
                    <h3>Steps to Reduce Disease:</h3> 
                    <p>{{ steps }}</p> 
                </div> 
            </div> 
        {% endif %} 
</body> 
8 | P a g e  
      </html> 
Style.css 
body { 
    font-family: 'Arial', sans-serif; 
    background-image: url('/static/images/paddy-field.jpg'); /* Background image of paddy fields */ 
    background-size: cover; 
    background-position: center; 
    background-repeat: no-repeat; 
    color: #4b0082; /* Dark purple for text */ 
    margin: 0; 
    padding: 0; 
    display: flex; 
    flex-direction: column; 
    align-items: center; 
    min-height: 100vh; 
} 
    padding: 10px 20px; 
    width: 100%; 
    display: flex; 
    justify-content: space-between; /* Space between left and right sections */ 
    align-items: center; 
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Shadow for depth */ 
} 
.nav-title { 
    color: #4b0082; /* Dark purple for the title */ 
    font-size: 24px; 
9 | P a g e  
    margin: 0; 
    font-weight: bold; /* Make the title bold */ 
} 
.nav-right { 
    display: flex; 
    gap: 10px; /* Space between buttons */ 
} 
.nav-button { 
    background-color: #8a2be2; /* Purple for buttons */ 
    color: white; 
    border: none; 
    padding: 10px 20px; 
    border-radius: 5px; 
    cursor: pointer; 
    font-size: 16px; 
    transition: background-color 0.3s ease, transform 0.3s ease; 
} 
.main-content { 
    width: 100%; 
    max-width: 800px; /* Limit the width for better readability */ 
    padding: 20px; 
    text-align: center; 
    background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white background */ 
    border-radius: 10px; 
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Shadow for depth */ 
    margin-top: 20px; /* Space below the navbar */ 
10 | P a g e  
    animation: fadeIn 1s ease-in-out; /* Fade-in animation */ 
} 
                        
Script.js 
// Function to upload image and get prediction 
async function uploadImage() { 
    const fileInput = document.getElementById("file-upload"); 
    const formData = new FormData(); 
    formData.append("file", fileInput.files[0]); 
    document.getElementById("loading-spinner").style.display = "block"; 
    try { 
        // Send the image to the backend 
        const response = await fetch("/predict", { 
            method: "POST", 
            body: formData, 
        }); 
        // Show success notification 
        showNotification("Image uploaded successfully!", "success"); 
    } catch (error) { 
        // Show error notification 
        showNotification("An error occurred. Please try again.", "error"); 
    } finally { 
        // Hide loading spinner 
        document.getElementById("loading-spinner").style.display = "none"; 
    } 
} 
11 | P a g e  
// Function to show medicine recommendations 
function showMedicine() { 
    document.getElementById("medicine").style.display = "block"; 
    document.getElementById("steps").style.display = "none"; 
} 
// Function to show steps to reduce disease 
function showSteps() { 
    document.getElementById("steps").style.display = "block"; 
    document.getElementById("medicine").style.display = "none"; 
} 
// Function to delete the uploaded image and reset the form 
function deleteImage() { 
    // Hide the result section 
    const resultSection = document.getElementById("result-section"); 
    if (resultSection) { 
        resultSection.style.display = "none"; 
    } 
    // Show notification 
    showNotification("Image deleted successfully.", "success"); 
} 
APP.PY 
 from flask import Flask, render_template, request, redirect, url_for 
import os 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
import numpy as np 
12 | P a g e  
app = Flask(__name__) 
# Load the trained model 
model = load_model("paddy_disease_model.h5") 
# Define class labels (replace with your actual disease names) 
class_labels = [ 
    "bacterial_leaf_blight", 
    "bacterial_leaf_streak", 
    "blast", 
] 
# Define medicine and steps for each disease 
disease_info = { 
    "bacterial_leaf_blight": { 
        "medicine": "Streptomycin or Copper-based fungicides.", 
        "steps": "Remove infected plants, avoid overwatering, and use resistant varieties." 
    }, 
} 
# Home page 
@app.route("/") 
def home(): 
    return render_template("index.html") 
# Handle image upload and prediction 
@app.route("/predict", methods=["POST"]) 
def predict(): 
    if "file" not in request.files: 
        return redirect(url_for("home")) 
    file = request.files["file"] 
13 | P a g e  
    if file.filename == "": 
        return redirect(url_for("home")) 
    # Preprocess the image for the model 
    img = image.load_img(image_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 
    # Make prediction 
    predictions = model.predict(img_array) 
    predicted_class = np.argmax(predictions, axis=1) 
    disease_name = class_labels[predicted_class[0]] 
    # Render the result page 
    return render_template( 
        "index.html", 
        disease=disease_name, 
        medicine=medicine, 
        steps=steps, 
        image_path=image_path 
    ) 
if __name__ == "__main__": 
    app.run(debug=True) 
Train_model.py 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
import os 
14 | P a g e  
import numpy as np 
from tensorflow.keras.preprocessing import image 
# Define paths 
train_dir = "dataset/train_images"  # Path to training images (with subfolders) 
test_dir = "dataset/test_images"    # Path to test images (no subfolders) 
img_size = (224, 224)               # Image size for the model 
batch_size = 32                     # Batch size for training 
# Data preprocessing and augmentation for training data 
train_datagen = ImageDataGenerator( 
    rescale=1.0 / 255.0,            # Normalize pixel values 
    rotation_range=20,               # Randomly rotate images 
    width_shift_range=0.2,           # Randomly shift images horizontally 
    height_shift_range=0.2,          # Randomly shift images vertically 
    horizontal_flip=True,            # Randomly flip images horizontally 
    validation_split=0.2             # Use 20% of data for validation 
) 
# Load training data 
train_generator = train_datagen.flow_from_directory( 
    train_dir, 
    target_size=img_size, 
    batch_size=batch_size, 
    class_mode="categorical", 
    subset="training"  # Use 80% of data for training 
) 
# Load validation data 
validation_generator = train_datagen.flow_from_directory( 
15 | P a g e  
    train_dir, 
    target_size=img_size, 
    batch_size=batch_size, 
    class_mode="categorical", 
    subset="validation"  # Use 20% of data for validation 
) 
# Load validation data 
validation_generator = train_datagen.flow_from_directory( 
    train_dir, 
    target_size=img_size, 
    batch_size=batch_size, 
    class_mode="categorical", 
    subset="validation"  # Use 20% of data for validation 
) 
# Define class labels (replace with your actual disease names) 
class_labels = list(train_generator.class_indices.keys()) 
print("Class Labels:", class_labels) 
# Build the CNN model 
model = Sequential([ 
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)), 
    MaxPooling2D((2, 2)), 
    Flatten(), 
    Dense(128, activation="relu"), 
    Dropout(0.5), 
    Dense(len(class_labels), activation="softmax")  # Output layer 
]) 
