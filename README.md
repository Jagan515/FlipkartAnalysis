# Smartphone Price Analysis and Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Completed-success)

This repository contains a comprehensive data science project focused on analyzing smartphone data scraped from Flipkart, one of India's leading e-commerce platforms. The project leverages web scraping, data cleaning, exploratory data analysis (EDA), machine learning, and interactive visualization to uncover market trends and predict smartphone prices. By combining Python's powerful libraries with a user-friendly dashboard, this project offers valuable insights for consumers, retailers, and tech enthusiasts. 

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Scope](#future-scope)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview
The Smartphone Price Analysis and Prediction project aims to understand pricing dynamics and consumer preferences in the Indian smartphone market. Using web scraping, the project collects detailed product information from Flipkart, including prices, specifications, ratings, and reviews. Through rigorous data cleaning and EDA, it identifies key trends, such as popular hardware configurations and brand dominance. Machine learning models, including Linear Regression and Random Forest, are employed to predict prices based on features like RAM, storage, and camera resolution. An interactive dashboard built with Dash provides a visual interface to explore these insights, making the project accessible to a wide audience.

**Objectives**:
1. Scrape and compile a robust dataset of smartphone listings.
2. Analyze price distributions and feature correlations to uncover market trends.
3. Build accurate price prediction models using machine learning.
4. Visualize findings through static plots and an interactive dashboard.
5. Provide actionable insights for stakeholders in the smartphone ecosystem.

## Features
- **Web Scraping**: Automated extraction of smartphone data from Flipkart using `requests` and `BeautifulSoup`, with user agent rotation to avoid detection.
- **Data Cleaning**: Standardized dataset with regex-based feature extraction (e.g., RAM, storage) and imputation of missing values.
- **Exploratory Data Analysis**:
  - Visualizations of price, rating, RAM, storage, and brand distributions.
  - Correlation analysis to identify price drivers.
  - Word cloud for qualitative insights from product descriptions.
- **Predictive Modeling**:
  - Linear Regression and Random Forest models for price prediction.
  - Hyperparameter tuning with GridSearchCV for optimized performance.
  - Feature importance analysis to highlight key predictors.
- **Interactive Dashboard**: A Dash-based web app for exploring price trends and battery-price relationships by brand.
- **Reproducible Workflow**: Modular scripts with clear documentation for easy replication.

## Dataset
The dataset is sourced via web scraping from Flipkart's smartphone search results (`https://www.flipkart.com/search?q=phones`). It includes:

- **Columns**: Product name, description, price, rating, reviews, RAM (GB), storage (GB), display size (inches), camera (MP), battery (mAh), warranty (years), and brand.
- **Size**: Approximately 1,000 unique smartphone listings across 59 pages.
- **Files**:
  - `flipkart_phones1.csv`: Raw scraped data.
  - `checkclean_final.xlsx`: Cleaned and processed dataset.
- **Note**: Due to ethical considerations, raw data is not shared publicly. Users can regenerate the dataset using the provided scraping script.

## Installation
To run this project locally, ensure you have Python 3.11+ installed. Follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Jagan515/FlipkartAnalysis
   cd FlipkartAnalysis
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Requirements File**:
   The `requirements.txt` includes:
   ```
   requests==2.31.0
   beautifulsoup4==4.12.2
   pandas==2.2.2
   numpy==1.26.4
   seaborn==0.13.2
   matplotlib==3.8.4
   scikit-learn==1.5.1
   wordcloud==1.9.3
   dash==2.17.1
   plotly==5.22.0
   ```

## Usage
1. **Web Scraping**:
   - Run `WebScrapping.py` to collect smartphone data from Flipkart:
     ```bash
     python WebScrapping.py
     ```
   - Output: `flipkart_phones1.csv`.
   - **Caution**: Adjust the sleep time (`time.sleep(45)`) to avoid server overload and comply with Flipkart’s terms of service.

2. **Data Cleaning and EDA**:
   - Execute `Final_python.py` to clean the data, perform EDA, and generate visualizations:
     ```bash
     python Final_python.py
     ```
   - Output: Cleaned dataset (`checkclean_final.xlsx`) and plots (saved locally if configured).

3. **Predictive Modeling**:
   - The `Final_python.py` script trains Linear Regression and Random Forest models, with results printed to the console (e.g., MSE, R² scores).

4. **Interactive Dashboard**:
   - Launch the Dash app to explore data visually:
     ```bash
     python Final_python.py
     ```
   - Access the dashboard at `http://127.0.0.1:8050` in your browser.
   - Use the brand dropdown to view price distributions and battery-price scatter plots.

5. **Sample Output**:
   - Visualizations: Price histograms, RAM/storage counts, correlation heatmaps, word clouds.
   - Model Performance: Random Forest R² ≈ 0.85 after tuning.
   - Dashboard: Interactive plots for brand-specific analysis.

## Project Structure
```
smartphone-price-analysis/
│
├── WebScrapping.py          # Web scraping script
├── Final_python.py         # Data cleaning, EDA, modeling, and dashboard
├── flipkart_phones.csv    # Raw scraped data (sample, not included)
├── checkclean_final.xlsx   # Cleaned dataset (sample, not included)
├── requirements.txt        # Python dependencies
├── plots/                  # Folder for saved visualizations (optional)
└── README.md               # Project documentation
```

## Results
- **EDA Insights**:
  - Most smartphones are priced between ₹10,000–₹30,000, with a peak at ₹15,000–₹20,000.
  - 8GB RAM and 128GB storage dominate, reflecting mid-range preferences.
  - Ratings cluster at 4.0–4.5 for budget phones, indicating high satisfaction.
  - RAM, storage, and camera resolution are key price drivers (correlation > 0.5).
  - Xiaomi, Realme, and Vivo lead the budget segment with 20–30 models each.
- **Modeling**:
  - Linear Regression: R² ≈ 0.6, MSE ≈ 1e8.
  - Random Forest (tuned): R² ≈ 0.85, MSE ≈ 5e7, with RAM and storage as top predictors.
- **Dashboard**: Enables brand-specific exploration of price trends and feature relationships.

## Future Scope
- **Multi-Platform Scraping**: Include Amazon and Snapdeal for a broader market view.
- **Sentiment Analysis**: Analyze reviews to gauge customer sentiment.
- **Advanced Models**: Test XGBoost or neural networks for better predictions.
- **Enhanced Dashboard**: Add filters for RAM, price, or 5G support.
- **Recommendation System**: Suggest phones based on user preferences and budget.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, reach out via:
- **GitHub Issues**: [Create an issue](https://github.com/Jagan515/FlipkartAnalysis/issues)
- **Email**: jaganp515@gmail.com 

Happy analyzing, and enjoy exploring the smartphone market!
