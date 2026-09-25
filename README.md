```markdown
# Advanced Topics in Machine Learning - Programming Assignment 1 (ATML PA1)

This repository contains the complete implementation, experimental results, and modular codebase for Programming Assignment 1.

---

## Overview & Execution

All experimental pipelines, domain adaptation methods, model training routines, and evaluations are fully executed and documented across the four primary Jupyter Notebooks[cite: 3]:

- **`ATML_PA1_task1_Complete.ipynb`**
- **`ATML_PA1_Task2_Complete.ipynb`**
- **`ATML_PA1_Task3_Complete.ipynb`**
- **`ATML_PA1_Task4_Complete.ipynb`**

Every notebook cell has been run end-to-end, with all logs, loss curves, and evaluation tables preserved inline for direct verification.

---

## Modular Repository Structure

To match the modular project structure suggested by the Teaching Assistants (TAs), the notebooks utilize `%%writefile` commands to automatically extract components into structured Python files and directories:


ATML_PA1/
│
├── ATML_PA1_task1_Complete.ipynb
├── ATML_PA1_Task2_Complete.ipynb
├── ATML_PA1_Task3_Complete.ipynb
├── ATML_PA1_Task4_Complete.ipynb
│
├── Task1/
│   ├── analysis/
│   ├── configs/
│   ├── data/
│   ├── models/
│   ├── results/
│   ├── scripts/
│   ├── README.md
│   └── __init__.py
│
├── Task2/
│   ├── evaluation/
│   ├── experiments/
│   ├── methods/
│   ├── models/
│   ├── results/
│   ├── __init__.py
│   ├── evaluate_cdan.py
│   ├── evaluate_dan.py
│   ├── evaluate_dann.py
│   ├── evaluate_final.py
│   ├── train.py
│   ├── train_cdan.py
│   ├── train_dan.py
│   └── train_dann.py
│
├── Task3/
│   ├── evaluation/
│   ├── methods/
│   ├── models/
│   ├── results/
│   ├── selection/
│   ├── __init__.py
│   ├── evaluate_sketch.py
│   └── train.py
│
├── task4/
│   ├── methods/
│   ├── models/
│   ├── results/
│   ├── __init__.py
│   ├── dataset_protocol.py
│   └── train_models.py
│
├── .gitignore
└── README.md

```

---

## Replication Guide (Step-by-Step)

Follow these exact steps to replicate the environment, reproduce the results, or run evaluations:

### Step 1: Clone Repository into Google Drive

Mount Google Drive and clone the repository directly to your workspace so all script extractions and checkpoint paths resolve seamlessly:

```python
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your preferred Drive directory
%cd /content/drive/MyDrive

# Clone repository
!git clone [https://github.com/Abdullah-61/ATML_PA1.git](https://github.com/Abdullah-61/ATML_PA1.git)
%cd /content/drive/MyDrive/ATML_PA1

```

### Step 2: Open the Desired Task Notebook

Open any of the target notebooks in Google Colab:

* For Task 1: Open `ATML_PA1_task1_Complete.ipynb`
* For Task 2: Open `ATML_PA1_Task2_Complete.ipynb`
* For Task 3: Open `ATML_PA1_Task3_Complete.ipynb`
* For Task 4: Open `ATML_PA1_Task4_Complete.ipynb`

Ensure your Colab runtime hardware accelerator is set to **GPU** (`Runtime` → `Change runtime type` → `T4 GPU`).

### Step 3: Run Setup and Script Generation Cells

1. Execute the initial configuration cells to verify GPU access and establish working paths.
2. Run the `%%writefile` cells sequentially. These write out the required `.py` modules, evaluation scripts, and utilities into the corresponding folders (`Task1/`, `Task2/`, `Task3/`, or `task4/`).



### Step 4: Model Execution and Checkpoint Usage

* **Fast Evaluation via Checkpoints:** If you want to skip long training runs, navigate to the evaluation section of the notebook. The evaluation routines point directly to pre-trained weights and checkpoints stored within each task directory.
* **Retraining from Scratch:** To train models from the ground up, run the training cells (e.g., executing `train.py`, `train_dann.py`, or `train_cdan.py` via notebook commands). Outputs and weights will automatically populate their respective `results/` or `models/` folders.



```

```
