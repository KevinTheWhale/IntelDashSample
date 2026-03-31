# IntelDash — Semiconductor Process Intelligence Dashboard

A production-style data dashboard built with **Plotly Dash** for analyzing semiconductor manufacturing process data using the [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom).

Built as a sample project demonstrating data engineering, SPC (Statistical Process Control), and manufacturing analytics relevant to semiconductor fab environments.

---

## Dashboard Features

- **KPI Cards** — Total wafers, yield rate, pass/fail counts, average null rate
- **Rolling Yield Trend** — 50-wafer rolling window yield % over production run
- **Pass/Fail Donut Chart** — Class distribution with yield % center label
- **Interactive Control Chart** — Per-feature X-bar chart with UCL/LCL ±3σ and out-of-control point detection
- **Null Rate Bar Chart** — Top 20 features ranked by missingness (color-coded)
- **Feature Distribution by Outcome** — Overlaid histogram comparing Pass vs Fail distributions per feature

---

## Dataset

**UCI SECOM Dataset** — Real semiconductor manufacturing process data  
- 1,567 wafer observations  
- 591 process parameters  
- Binary labels: -1 (Pass), +1 (Fail)  
- ~93% pass rate (class imbalanced)

Download from: https://archive.ics.uci.edu/dataset/179/secom  
Place `uci-secom.csv` in the project root directory.

> The dataset is not included in this repository. Download it separately.

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/KevinTheWhale/IntelDashSample.git
cd IntelDashSample
```

### 2. Create and activate virtual environment
```bash
python -m venv IntelDash
source IntelDash/bin/activate  # Mac/Linux
IntelDash\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download `uci-secom.csv` and place it in the project root.

### 5. Run the dashboard
```bash
python IntelDash.py
```

Open your browser at: `http://localhost:8050`

---

## Tech Stack

- Python
- Plotly Dash
- Pandas / NumPy
- Scikit-learn
- UCI SECOM Dataset

---

## Author

Kevin H. Pham  
M.S. Mechanical Engineering (In Progress) · M.S. Statistics · B.S. Applied Computational Mathematics  
San Jose State University
