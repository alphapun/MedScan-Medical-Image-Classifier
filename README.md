# MedScan — Multi-Agent Diagnostic AI

## Overview
MedScan AI is a Multi-Agent Diagnostic AI application designed to assist in medical image analysis. It leverages deep learning models to classify pathologies across multiple medical domains and provides an automated, explainable clinical reporting pipeline.

The repository contains both the research phase (model training and evaluation) and the production phase (an interactive web application).

## Project Structure

### 1. Web Application (`app.py`)
An interactive Streamlit frontend built around an agentic diagnostic loop:
* **Vision Agent**: Runs local inference on medical images using fine-tuned architectures (ConvNeXt, ViT, EfficientNet) and generates Grad-CAM explainability heatmaps to highlight regions of interest.
* **Diagnostic Agent**: Starts automatically on upload. Gemini 2.5 Flash acts as the planner. It sees the scan and decides which tools to call and when to stop; every result on the page comes from what it did:
  * `verify_scan(detected_type, image_quality_ok, reason)` must be called first. The agent decides from the image whether it is an OCT, brain MRI or chest X-ray (the sidebar dropdown is only the expected type) or rejects it. The code refuses to run any model on a scan that has not been verified as that domain.
  * `classify_scan(domain)` runs the primary model for the verified domain.
  * `generate_heatmap(domain, target_class)` makes a Grad-CAM heatmap for the classes the agent chooses to explain (e.g. both of the top two when they are close).
  * `second_opinion(model)` re-classifies OCT scans with the independently trained ViT-B/16, DeiT3-B or EfficientNetV2-S models.
  * `class_reliability(domain)` returns each model's per-class test recall, so the agent knows how far to trust a prediction (e.g. DRUSEN is the weakest class).
  * A separate Gemini call then **self-checks** the report against the tool results, and the agent revises it if needed.
  * **Guardrails in code** (not left to the LLM): a cap of 10 tool calls, verification before any model runs, no report or PDF unless a classifier ran, a forced warning on model disagreement or confidence below 80%, and an always-present disclaimer.
  * The pipeline log shows the tools the agent actually chose, so an easy scan takes fewer steps than a borderline one.
* **Document Agent**: Compiles the report and heatmap into a downloadable PDF.

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
   pip install streamlit torch torchvision timm pillow numpy plotly google-genai fpdf2 grad-cam opencv-python
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
* `vit_best.pth`, `deit_best.pth`, `cnn_best.pth` (optional OCT second-opinion models, trained in `Experiments.ipynb`). Without them the agent's `second_opinion` tool reports that the weights are unavailable.

---
**Disclaimer:** *This tool is for research and educational purposes only. It is not a medical device. Clinical decisions must be made by qualified healthcare professionals.*
