# MedScan — Multi-Agent Diagnostic AI

## Overview
MedScan AI is a Multi-Agent Diagnostic AI application designed to assist in medical image analysis. It leverages deep learning models to classify pathologies across multiple medical domains and provides an automated, explainable clinical reporting pipeline.

The repository contains both the research phase (model training and evaluation) and the production phase (an interactive web application).

## Project Structure

### 1. Web Application (`app.py`)
An interactive Streamlit frontend that orchestrates a multi-agent diagnostic pipeline:
* **Vision Agent**: Runs local inference on medical images using fine-tuned architectures (ConvNeXt, ViT, EfficientNet) and generates Grad-CAM explainability heatmaps to highlight regions of interest.
* **Report Agent**: Utilizes Google's Gemini 2.5 Flash LLM to generate structured, professional medical reports based on the Vision Agent's findings and confidence levels.
* **Document Agent**: Compiles the clinical findings and heatmaps into a downloadable PDF report.

**Supported Domains:**
* **Retinal OCT (Eye):** CNV, DME, Drusen, Normal
* **Brain Tumour (MRI):** Glioma, Meningioma, Pituitary, No Tumor
* **Lung Pathology (X-Ray):** 15-class classification including Pneumonia, Effusion, Mass, etc.

### 2. Model Training & Experiments (`Experiments.ipynb`)
A comprehensive Jupyter Notebook detailing the research, training, and evaluation workflow for the Retinal OCT models.
* **Data Processing**: Custom PyTorch Dataset/DataLoader pipelines for Supervisely JSON annotations.
* **Training Techniques**: Layer-wise Learning Rate Decay (LLRD) for Vision Transformers (ViT, DeiT3), Cosine Annealing scheduling, and label smoothing.
* **Monitoring**: Integrated Telegram bot notifications for live epoch tracking and early stopping.
* **Evaluation**: Extensive metric comparisons (Accuracy, Macro F1, Class-wise Recall, AUC) and Confusion Matrix generation across CNN Baseline, ViT-B/16, DeiT3-B, and ConvNeXt. Outputs are logged to the `Eval/` and `logs/` directories.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Install dependencies:**
   Ensure you have Python 3.8+ installed.
   ```bash
   pip install streamlit torch torchvision timm pillow numpy plotly google-generativeai fpdf grad-cam opencv-python
   ```

3. **Set your API Key:**
   The Report Agent requires a Gemini API key. Set it as an environment variable:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Model Weights
*(Note: Due to GitHub's file size limits, the trained model weights are not stored directly in this repository.)*

To run the application locally, please download the model weights and place them in the project root directory:
* `convnext_best.pth` (Retinal OCT) - [[Link to download](https://huggingface.co/alphapun/convnext_best)]
* `brain_best.pt` (Brain MRI) - [[Link to download](https://huggingface.co/Jaubert-Wang/MRI_Brain_Cancer_Classification_Deeplearning)]
* `chest_best.pt` (Chest X-Ray) - [[Link to download](https://huggingface.co/Beiry/medscan-chest-xray14)]

---
**Disclaimer:** *This tool is for research and educational purposes only. It is not a medical device. Clinical decisions must be made by qualified healthcare professionals.*
