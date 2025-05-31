# DATA3888 Group 4


## 🛠 Environment Setup

### Option A — Using Conda 

```bash
conda create -n claropath python=3.9
conda activate claropath
pip install -r requirements.txt
```


### Option B — Using venv and pip
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt 
```


## 📂 Project Files Required
```bash 
├── Shiny_model.py              # main Streamlit app  
├── saved_model/  
│   └── best_swin.pth           # trained Swin Transformer weights  
├── 41467_2023_43458_MOESM4_ESM.xlsx   # metadata file  
├── all_items.txt               # paths to annotated image files                      
├── cbr.csv                     # cell coordinates for spatial layout  
├── requirements.txt            # Python dependencies  
└── README.md                   # this file
```
## ▶️ Running the App
```bash 
streamlit run Shiny_model.py
```
