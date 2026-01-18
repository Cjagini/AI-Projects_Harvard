# Traffic Sign Recognition Experimentation

## The Goal

The objective was to build a Convolutional Neural Network (CNN) using TensorFlow to classify 43 different types of road signs with high accuracy.

## The Process

1. **Initial Model:** I started with a basic setup: one convolutional layer and one max-pooling layer. Early results showed low accuracy (~5%), meaning the model was too "simple" to see the complex differences between signs.
2. **Adding Depth:** I added a second and third convolutional layer. This allowed the AI to detect finer details (like the difference between a "20" and "30" speed limit sign).
3. **Preventing Overfitting:** I implemented a Dropout layer (0.5). This was vital because, without it, the model would memorize the training data and fail on new images.
4. **Optimizing Dense Layers:** Increasing the hidden layer size to 256 units provided the "brain power" needed to categorize all 43 classes.

## Results

Final Accuracy: 0.9690
Final Loss: 0.1317
The model is now robust enough to identify traffic signs with over 96% reliability.
