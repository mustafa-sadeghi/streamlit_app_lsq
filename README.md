# Interactive Regression Visualization with Streamlit and Plotly

## Description

A fully interactive regression visualization tool built with **Streamlit** and **Plotly**.  
This app demonstrates different ways of measuring distances between data points and a fitted regression line, including **vertical**, **horizontal**, and **perpendicular** distances.  
The tool dynamically updates the plot as the user adjusts the **slope**, **intercept**, and **distance type**, making it ideal for learning regression concepts or teaching error metrics and least-squares intuition.

---

## Features

- Adjustable **slope** and **intercept** using Streamlit sliders  
- Visualization of:
  - Vertical residuals  
  - Horizontal residuals  
  - Perpendicular distances to the regression line  
- Automatic computation of **Sum of Squared Distances (SSD)** for each method  
- Real-time Plotly rendering with smooth UI  
- Clean code structure with reusable functions

---

## Project Structure

```text
.
├── app.py                     # Main Streamlit app
├── requirements.txt (optional)
└── README.md
```

## Installation
Install required dependencies:
```bash
pip install streamlit plotly numpy
```

## Running the App
Inside the project folder, run:
```bash
streamlit run app.py
```
Then open the displayed local URL in your browser (e.g., http://localhost:8501)

## How It Works

- The script generates synthetic linear data:  
  **y = 0.5x + noise**
- The user interacts with:
  - **Line slope**
  - **Line intercept**
  - **Distance type**: vertical, horizontal, perpendicular
- For each data point:
  - The corresponding distance line is drawn  
  - **SSD** (sum of squared distances) updates live in the chart title
- **Plotly** renders the interactive visualization  
- **Streamlit** updates controls and UI elements

---

## Example Use Cases

- Teaching **least squares regression** concepts  
- Visualizing the meaning of **residuals**  
- Comparing **vertical**, **horizontal**, and **perpendicular** distances  
- Demonstrating geometric interpretation of regression lines  
- Creating intuitive demos for **ML** and **statistics** education  
