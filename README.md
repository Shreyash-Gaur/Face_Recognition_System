
---

# Face Recognition System

This project builds a face recognition system, drawing inspiration from the FaceNet model and incorporating the power of the triplet loss function. It aims to achieve two primary goals:

1. **Face Verification:** Determining if two given images are of the same person (1:1 matching).
2. **Face Recognition:** Identifying an individual from a set of many (1:K matching).

## Key Concepts & Approach

* **FaceNet Inspiration:** The project draws heavily from the FaceNet architecture, which has proven effective for face recognition tasks.
* **128-Dimensional Embeddings:** The system encodes face images into compact 128-dimensional vectors, capturing essential facial features.
* **Triplet Loss:** The core concept behind the training of this model is the *triplet loss function*, which ensures:
  * Minimization of the distance between encodings of the same person (Anchor and Positive images).
  * Maximization of the distance between encodings of different people (Anchor and Negative images).

## Project Structure

*  `Face Recognition.ipynb`: Contains core logic for building and using the face recognition model.
* `inception_blocks_v2.py`: Defines the Inception model architecture used for face encoding.
* `fr_utils.py`: Contains utility functions for loading weights and datasets.
* `weights/`: Directory containing pre-trained model weights.
* `datasets/`: Directory containing training and testing datasets.
* `images/`: Directory containing sample images for recognition.
* `models/`: Directory to save the trained model.

## Implementation Steps

1. **Environment Preparation:**

   * **Install Necessary Packages:** Make sure you have the required libraries installed: Keras (with TensorFlow backend), OpenCV (cv2), NumPy, Pandas, etc.
   * **Import Modules:** Import all essential modules and set Keras to use 'channels_first' image data format.

2. **Triplet Loss Function:**

   * **Understand the Formula:** Deeply study the triplet loss formula. It aims to minimize the distance between the anchor image encoding and the positive image encoding, while maximizing the distance between the anchor image encoding and the negative image encoding.
   * **Code Implementation:** Follow the provided code snippet and comments to implement the `triplet_loss` function. Use TensorFlow operations like `tf.reduce_sum`, `tf.square`, `tf.subtract`, `tf.add`, and `tf.maximum`. Pay attention to the axis parameter in `tf.reduce_sum` to ensure proper computation along the desired dimensions.

4. **Load Pre-trained Model:**

   * **Model Architecture:** Understand that we use a pre-trained FaceNet model based on the Inception architecture. You can examine the `inception_blocks_v2.py` file for more details on its implementation.
   * **Load Weights:** Use the `load_weights_from_FaceNet` and `FRmodel.compile` functions to load the pre-trained weights into your Keras model. This saves significant training time and leverages existing knowledge.

5. **Build the Face Encoding Database:**

   * **Image Preprocessing:** Grasp the `img_to_encoding` function. It loads an image, converts it to RGB format, normalizes pixel values, and feeds it to the model to get the 128-dimensional encoding.
   * **Create the Database:**  Build a Python dictionary (`database`) where keys are names (strings) and values are the corresponding face encodings (128-dimensional vectors) obtained using the `img_to_encoding` function.

6. **Implement Face Verification:**

   * **`verify` Function:** Study the `verify` function. It takes an image path, a claimed identity, the database, and the model as input.
   * **Encoding Comparison:** It computes the encoding of the input image and compares it (using L2 distance) with the encoding of the claimed identity from the database.
   * **Access Decision:** If the distance is below a threshold (0.7 in this case), it grants access; otherwise, it denies access.

7. **Test Face Verification:**

   * **Positive and Negative Cases:** Run the `verify` function with sample images representing both positive (authorized person) and negative (unauthorized person) cases to observe the system's behavior.

8. **Implement Face Recognition:**

   * **`who_is_it` Function:** Examine the `who_is_it` function. It takes an image path, the database, and the model as input.
   * **Closest Match Search:**  It computes the encoding of the input image and iterates through the database to find the encoding with the smallest L2 distance.
   * **Identity Prediction:** If the minimum distance is below the threshold, it predicts the identity associated with the closest encoding; otherwise, it indicates that the person is not in the database.

9. **Test Face Recognition:**

   * **Identify Individuals:**  Test the `who_is_it` function with different images to see if it correctly recognizes individuals in the database.

10. **Explore Improvements:**

    * **Data Augmentation:**  Consider adding more images per person with variations in lighting, pose, and expressions to improve robustness.
    * **Face Detection and Cropping:** Preprocess images to detect and crop faces, focusing the model on the relevant region and potentially improving accuracy.
    * **Advanced Models:** Explore newer and more sophisticated face recognition models that may offer better performance.
    * **Ethical Considerations:**  Be mindful of potential biases in training data and the ethical implications of face recognition technology.

10. **Additional Considerations:**

    * **Real-world Challenges:**  Understand that real-world face recognition systems often encounter challenges due to variations in lighting, pose, facial expressions, occlusions, and image quality.
    * **Scalability:** If you plan to deploy a face recognition system at a larger scale, consider optimizations for efficient database search and handling a large number of identities.
    * **Security:**  Implement appropriate security measures to protect the face encoding database and prevent unauthorized access or manipulation.

## Conclusion

This detailed guide has walked you through the core concepts and implementation steps involved in building a basic face recognition system using triplet loss. Understanding the underlying principles and experimenting with the code can lay a strong foundation for further exploration and development in this exciting field. Remember that continuous learning and staying updated with the latest advancements in face recognition technology are crucial for building effective and responsible systems.

## How to Run

1. **Clone the repository:** `git clone https://github.com/Shreyash-Gaur/Face_Recognition_System.git`
2. **Prepare datasets:** Organize your images in the `images/` directory & edit the database at 3.1 - Face Verification.
3. **Run the code:** Execute the provided code snippets in a Jupyter Notebook.

## Contributing

Feel free to contribute by opening issues or pull requests.

## License

This project is licensed under the MIT License.

---


