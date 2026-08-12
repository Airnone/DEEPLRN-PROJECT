# [Project Name]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MC40mm4NnUlgzZ9C8X_XhResio9s9cL8?usp=sharing)

## Overview
DEEPLRN is an automated Natural Language Processing (NLP) pipeline built to extract and structure information from the Philippine Commission on Audit (COA) Annual Audit Reports. 

Because these reports consist of hundreds of pages of unstructured text, it is extremely time-consuming to manually search for specific financial or compliance anomalies. This project solves that by automatically reading the text and performing three main tasks:
1. **Entity Recognition:** Highlights key information like dates, monetary amounts, and organizations.
2. **Finding Classification:** Categorizes the overall audit finding into one of 9 specific types (e.g., *financial_reporting_accuracy* or *unliquidated_cash_advance*).
3. **Relation Extraction:** Links responsible organizations directly to the monetary amounts they are associated with.

The pipeline is built using PyTorch and Hugging Face Transformers. It uses a RoBERTa-base encoder combined with a document-level Transformer to handle the massive length of these reports without losing context, while keeping track of the exact character spans and page numbers for easy verification.

## Directory Structure
To run this project correctly, ensure your local environment or Google Drive (if using Colab) matches the following folder structure:

```text
deeplrn/
│
├── annotations/        # 
├── data/               # 
└── runs/               # 
```

## How to Run
This project is designed to run seamlessly in Google Colab. You do not need to install or configure anything on your local computer.

#### Prepare your Google Drive:
Create a folder named deeplrn in your root Google Drive (MyDrive/deeplrn). Upload the required annotations/ and data/ folders into this directory.

#### Open the Notebook:
Click the "Open In Colab" badge at the top of this page to open the main project notebook.

#### Save a Copy:
In Colab, go to File > Save a copy in Drive to create your own editable version of the notebook.

#### Enable GPU:
Ensure you are using a GPU runtime by clicking Runtime > Change runtime type and selecting GPU (e.g., T4 GPU).

#### Run the Pipeline:
Run the cells in order from top to bottom. The notebook is fully self-contained—it will automatically mount your Google Drive, write out the necessary package files, install the required dependencies, and execute the data splitting, training, and evaluation workflows.

