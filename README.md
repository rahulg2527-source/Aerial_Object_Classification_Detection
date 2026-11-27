\# Aerial Object Classification \& Detection

\### Bird vs Drone Recognition Using Deep Learning



An end-to-end computer vision project that combines image classification and object detection to identify birds and drones in aerial images.



The system uses a custom Convolutional Neural Network (CNN), MobileNetV2 transfer learning, and YOLOv8, supported by an interactive Streamlit application for image inference and model evaluation.



\## 1. Project Overview



This project implements two computer vision tasks:



\- \*\*Image Classification:\*\* Classifies an image as either Bird or Drone using MobileNetV2.

\- \*\*Object Detection:\*\* Detects and localizes birds and drones in images using YOLOv8.



A custom CNN is also implemented as a classification baseline.



\## 2. Key Features



\- Bird vs Drone image classification.

\- Custom CNN and MobileNetV2 models.

\- YOLOv8 object detection.

\- Bounding-box visualization and confidence scores.

\- Interactive Streamlit dashboard.

\- Model evaluation and classification reports.

\- Confusion matrices and dataset visualizations.

\- GPU acceleration when supported.



\## 3. Technology Stack



| Category | Technologies |

|---|---|

| Language | Python |

| Deep Learning | PyTorch, Torchvision |

| Classification | Custom CNN, MobileNetV2 |

| Object Detection | Ultralytics YOLOv8 |

| Image Processing | Pillow |

| Data Analysis | Pandas, NumPy |

| Visualization | Matplotlib, Seaborn |

| Evaluation | Scikit-learn |

| Application | Streamlit |



\## 4. Model Performance



\### MobileNetV2 — Image Classification



The MobileNetV2 model uses transfer learning with a custom two-class classification head.



| Metric | Test Result |

|---|---:|

| Accuracy | 98.14% |

| Bird Precision | 96.80% |

| Bird Recall | 100.00% |

| Bird F1-score | 98.37% |

| Drone Precision | 100.00% |

| Drone Recall | 95.74% |

| Drone F1-score | 97.83% |



\### YOLOv8n — Object Detection



YOLOv8n was trained for 50 epochs with an input image size of 416 × 416.



| Metric | Test Result |

|---|---:|

| Precision | 94.09% |

| Recall | 85.72% |

| mAP@50 | 91.30% |

| mAP@50-95 | 58.69% |



These are separate classification and detection evaluations and should not be interpreted as directly comparable metrics.



\## 5. Dataset



\### Classification Dataset



| Split | Images |

|---|---:|

| Training | 2,662 |

| Validation | 442 |

| Testing | 215 |

| Total | 3,319 |



Expected structure:



```text

data/raw/classification/

├── train/

│   ├── bird/

│   └── drone/

├── valid/

│   ├── bird/

│   └── drone/

└── test/

&#x20;   ├── bird/

&#x20;   └── drone/

```



\### Detection Dataset



| Split | Images |

|---|---:|

| Training | 2,622 |

| Validation | 330 |

| Testing | 327 |

| Total | 3,279 |



Expected structure:



```text

data/raw/detection/dataset/

├── train/

│   ├── images/

│   └── labels/

├── valid/

│   ├── images/

│   └── labels/

├── test/

│   ├── images/

│   └── labels/

└── data.yaml

```



\*\*Dataset note:\*\* Dataset files are not included in this repository. Obtain them from their authorized source and arrange them in the structures above.



\## 6. Project Structure



```text

Aerial\_Object\_Classification\_Detection/

│

├── app/

│   ├── streamlit\_app.py

│   ├── test\_classification.py

│   ├── test\_detection.py

│   └── utils/

│       ├── classification.py

│       └── detection.py

│

├── data/

│

├── outputs/

│   ├── plots/

│   └── reports/

│

├── src/

│   ├── data/

│   │   ├── eda.py

│   │   └── preprocessing.py

│   │

│   ├── models/

│   │   ├── custom\_cnn.py

│   │   └── mobilenet\_model.py

│   │

│   ├── evaluate.py

│   ├── evaluate\_mobilenet.py

│   ├── evaluate\_yolo.py

│   ├── predict\_yolo.py

│   ├── train.py

│   ├── train\_mobilenet.py

│   └── train\_yolo.py

│

├── .gitignore

├── requirements.txt

└── README.md

```



\## 7. Installation



\### Clone the repository



After creating the GitHub repository, replace the URL below with its actual clone URL.



```bash

git clone YOUR\_REPOSITORY\_URL

cd Aerial\_Object\_Classification\_Detection

```



\### Create a virtual environment



```bash

python -m venv .venv

```



Activate on Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\### Install dependencies



```bash

python -m pip install --upgrade pip

pip install -r requirements.txt

```



For NVIDIA GPU acceleration, install the PyTorch and Torchvision builds compatible with your hardware and CUDA environment. See the official \[PyTorch installation guide](https://pytorch.org/get-started/locally/).



\## 8. Model Checkpoints



Trained model weights are excluded from this repository because of their file sizes.



Expected checkpoint locations:



```text

outputs/models/best\_mobilenet\_v2.pth



outputs/yolo/bird\_drone\_yolov8n/weights/best.pt

```



You must obtain the trained checkpoints or train the models before running inference.



\## 9. Run the Application



From the project root:



```bash

streamlit run app/streamlit\_app.py

```



Open the local URL displayed in your terminal.



The application includes project overview, classification, object detection, and model performance sections.



\## 10. Training



Run the corresponding training scripts from the project root:



\*\*Custom CNN\*\*



```bash

python src/train.py

```



\*\*MobileNetV2\*\*



```bash

python src/train\_mobilenet.py

```



\*\*YOLOv8\*\*



```bash

python src/train\_yolo.py

```



Training requires the relevant datasets to be available in the expected locations.



\## 11. Evaluation



```bash

python src/evaluate.py

python src/evaluate\_mobilenet.py

python src/evaluate\_yolo.py

```



Evaluation reports and visualizations are saved under `outputs/`.



\## 12. Limitations



\- Classification accuracy reflects performance on the project's test split.

\- YOLOv8 recall is lower than precision, meaning some objects may be missed.

\- The mAP@50-95 result indicates scope for improving bounding-box localization.

\- Performance on unseen real-world aerial footage may differ.

\- This project is intended for experimentation and demonstration, not safety-critical deployment.



\## 13. Future Improvements



\- Expand training data with more diverse aerial conditions.

\- Improve detection recall and localization.

\- Evaluate performance across lighting conditions and object distances.

\- Experiment with larger detection architectures.

\- Add video and real-time camera inference.

\- Optimize models for edge deployment.



\## 14. Author



\*\*Rahul Gupta\*\*



\- GitHub: https://github.com/rahulg2527

\- LinkedIn: https://linkedin.com/in/rahul-gupta-65a763220



\## 15. License



No license has been selected for this repository yet. Add an appropriate license before allowing others to reuse or redistribute the project.

