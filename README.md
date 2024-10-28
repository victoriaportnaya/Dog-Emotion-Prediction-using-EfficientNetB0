# Dog Emotion Prediction using EfficientNetB0

This project aims to predict the emotional state of dogs—**angry**, **happy**, **relaxed**, or **sad**—from images using transfer learning with the EfficientNetB0 model.

## File Structure

```
├── DogEmotions.ipynb    # Jupyter notebook containing the entire workflow
├── requirements.txt     # List of required libraries
```

## Description

The project involves:

- **Data Preparation**: Loading and preprocessing images of dogs categorized into different emotions.
- **Data Augmentation**: Applying transformations such as rotation, zoom, shear, brightness adjustments, and flips to enhance the dataset and prevent overfitting.
- **Model Architecture**:
  - Utilizing **EfficientNetB0** as the base model with pre-trained ImageNet weights.
  - Adding custom layers on top of the base model for classification.
- **Training Process**:
  - Initial training with the base model layers frozen to train only the top layers.
  - Fine-tuning by unfreezing some layers of the base model and training with a lower learning rate.
  - Implementing early stopping and learning rate scheduling to optimize training.
- **Evaluation**:
  - Assessing the model's performance on the validation set.
  - Plotting training and validation accuracy and loss over epochs.
- **Visualization**:
  - Displaying sample images from each emotion category to understand the data distribution.

## Dataset

- **Data Directory**: You need to set the `data_dir` variable in the notebook to point to your dataset directory.
- **Expected Structure**:

  ```
  dataset/
  ├── angry/
  ├── happy/
  ├── relaxed/
  └── sad/
  ```

- Each subdirectory should contain images corresponding to that emotion category.

## Usage

### 1. Install Dependencies

Ensure you have the required packages installed. You can install them using the `requirements.txt` file provided in the repository:

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

- Organize your dataset as per the expected structure mentioned above.
- Make sure you have a sufficient number of images in each category for effective training.

### 3. Update the Data Directory Path

In the `DogEmotions.ipynb` notebook, update the `data_dir` variable to point to your dataset directory:

```python
data_dir = '/path/to/your/dataset'
```

### 4. Run the Jupyter Notebook

Launch Jupyter Notebook and open `DogEmotions.ipynb`:

```bash
jupyter notebook DogEmotions.ipynb
```

- Execute the cells sequentially to run the entire workflow.
- The notebook includes code for data preprocessing, model training, evaluation, and visualization.

### 5. Customize Parameters (Optional)

- **Data Augmentation**: You can adjust the parameters in the `ImageDataGenerator` to suit your dataset.
- **Model Configuration**: Modify the model architecture or training parameters like batch size, number of epochs, learning rate, etc., as needed.

### 6. Training the Model

- The notebook first trains the model with the base layers frozen.
- It then fine-tunes the model by unfreezing some layers and training with a lower learning rate.

### 7. Evaluating the Model

- After training, the model is evaluated on the validation set.
- Training and validation accuracy and loss are plotted for analysis.

### 8. Visualizing Sample Images

- The notebook includes a function to display sample images from each category.
- This helps in understanding the dataset and verifying that the images are loaded correctly.

## Notes

- **Class Mode**: Since the dataset has four categories, ensure that `class_mode` is set to `'categorical'` in the data generators:

  ```python
  class_mode='categorical'
  ```

- **Output Layer Adjustment**: Modify the output layer of the model to have four units with a `'softmax'` activation function:

  ```python
  predictions = Dense(4, activation='softmax')(x)
  ```

- **Loss Function**: Use `'categorical_crossentropy'` as the loss function when compiling the model for multi-class classification.

  ```python
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  ```

- **Preventing Overfitting**:

  - Utilize techniques like dropout, data augmentation, and regularization.
  - Monitor validation metrics to detect overfitting early.

- **Class Weights**: If your dataset is imbalanced, consider using class weights to give more importance to minority classes.

## Requirements

The `requirements.txt` file is already included in the repository. It contains all the necessary packages for this project.

Install the requirements using:

```bash
pip install -r requirements.txt
```

## Example Code Adjustments

Below are key code adjustments to handle multi-class classification:

### Data Generators

```python
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',  # Changed from 'binary' to 'categorical'
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',  # Changed from 'binary' to 'categorical'
    subset='validation'
)
```

### Model Output Layer

```python
# Adjust the output layer for 4 classes
predictions = Dense(4, activation='softmax')(x)
```

### Model Compilation

```python
optimizer = Adam(learning_rate=1e-5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```
