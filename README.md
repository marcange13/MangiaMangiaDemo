# 🍝 Mangia Mangia – AI Restaurant Menu Optimizer

Mangia Mangia is an AI-powered solution to help restaurants reduce food waste, optimize inventory, and generate high-profit daily specials. It combines machine learning, NLP, and pairing heuristics to create actionable menu recommendations based on real-time stock data.

## 🚀 Features

* **Inventory-Based Recommendations** – Suggests dishes using available ingredients from your live inventory.
* **Viability Scoring** – Weighted scoring system based on urgency, surplus, and profit margin for each dish.
* **Dynamic Menu Generator** – NLP-driven ingredient clustering to create new, creative dish ideas.
* **Food & Wine Pairing** – Automated ingredient and wine pairing suggestions using clustering + heuristics.
* **Tasty API Mock Integration** – Built-in simulation for recipe and cooking video suggestions from Tasty.
* **Demand Forecasting** – Predicts which dishes are likely to be in demand based on historical and seasonal trends.
* **Streamlit Front-End** – Fully interactive interface to browse recommendations and specials.

## 📂 Project Structure

```
Mangia-Mangia/
│── app.py                 # Streamlit front-end
│── menu_with_viability.csv # Menu dataset with scoring
│── authentic_inventory.csv # Restaurant inventory
│── requirements.txt        # Dependencies
│── README.md               # Project documentation
│── notebooks/              # Kaggle development notebooks
```

## ▶️ How to Run

1. Clone the repository
2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
3. Run the app:

   ```
   streamlit run app.py
   ```
4. Open the local URL to interact with the app.

---


