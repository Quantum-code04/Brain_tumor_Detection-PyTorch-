dowload dataset - https://www.kaggle.com/datasets/helloworldnamashte/brain-tumor-dataset


# Brain_tumor_Detection-PyTorch-
🧠 Brain Tumor Classification using ResNet18 / ResNet50 .   
This project is a Deep Learning-based Image Classification System to detect whether a brain MRI scan shows a Brain Tumor or is Healthy.
We use a pretrained ResNet18 model from PyTorch, fine-tuned on our custom dataset.

```
## 📂 Project Structure

Brain_Tumor_Classification/
│── Brain_Tumor_Data_Set/       # Dataset (two folders: Brain Tumor, Healthy)
│   ├── Brain Tumor/            # MRI images with tumors
│   ├── Healthy/                # MRI images without tumors
│
│── best_frozen_resnet18.pt     # Saved model with best performance (frozen layers)
│
│
│── Detection_Model.ipynb       # All python code.
│
│── README.md                   # Project documentation
```

## 🚀 Technologies Used

Python 3.9+
PyTorch → for deep learning model (ResNet18)
Torchvision → pretrained models, transforms
Matplotlib & Seaborn → visualization
sk-learn 
Pathlib & OS → dataset management


## 🧑‍💻 Model

We use ResNet18 (Residual Network with 18 layers) pretrained on ImageNet.
Key steps:
Freeze initial layers (transfer learning).
Replace the last fully connected layer for binary classification (Tumor vs Healthy).
Train on our dataset with CrossEntropy Loss + Adam optimizer.
Track Accuracy & F1-score.
Save best performing weights → best_frozen_resnet18.pt.

## 📊 Training

Epochs: 10 (configurable)
Optimizer: Adam
Loss: CrossEntropyLoss
Metrics: Accuracy & F1 Score
Early checkpoint saving when best F1 is found.

## 📈 Results

Achieved high accuracy and F1-score on validation set.
Can correctly classify Brain Tumor vs Healthy with good confidence.


##⚡ How to Run

### Clone the repo:

git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification


Install dependencies:
pip install torch torchvision matplotlib os 


## 📌 Future Improvements

Try deeper models (ResNet34, EfficientNet).

Add Grad-CAM visualization to highlight tumor regions.








