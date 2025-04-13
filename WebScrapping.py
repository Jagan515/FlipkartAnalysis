import requests as rq
from bs4 import BeautifulSoup as bp
import pandas as pd
import time
import urllib3
import random

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 🔁 User agent list for rotation
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.90 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
]

# Function to return a random user-agent
def get_random_headers():
    return {"User-Agent": random.choice(user_agents)}

# Lists to store product details
names = []
descriptions = []
reviews = []
prices = []
ratings = []

for page in range(1, 60):
    #url = f"https://www.flipkart.com/search?q=phones+under+50000&page={page}"
    url=f"https://www.flipkart.com/search?q=phones&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off&sort=recency_desc&page={page}"
    
    headers = get_random_headers()
    r = rq.get(url, headers=headers, verify=False)
    print(f"🔄 Page {page} | Status: {r.status_code} | Agent: {headers['User-Agent']}")

    if r.status_code != 200:
        print("❌ Failed to fetch page. Skipping...")
        continue

    soup = bp(r.text, "lxml")
    
    products = soup.find_all("div", class_="tUxRFH")  # This might need updating based on page source

    print(f"🔹 Found {len(products)} products on Page {page}")

    for item in products:
        name = item.find("div", class_="KzDlHZ")
        desc = item.find("div", class_="_6NESgJ")
        review = item.find("span", class_="Wphh3N")
        price = item.find("div", class_="Nx9bqj _4b5DiR")
        rating = item.find("div", class_="XQDdHH")

        names.append(name.text if name else "N/A")
        descriptions.append(desc.text if desc else "N/A")
        reviews.append(review.text if review else "N/A")
        prices.append(price.text if price else "N/A")
        ratings.append(rating.text if rating else "N/A")

    print(f"✅ Total collected: {len(names)} products")

    # Delay to reduce detection risk
    time.sleep(45)  # You can increase if you still get blocked

# Save to CSV
df = pd.DataFrame({
    "Product": names,
    "Description": descriptions,
    "Review": reviews,
    "Rating": ratings,
    "Price": prices
})
df.to_csv("flipkart_phones1.csv", index=False)
print("📦 Data saved to 'flipkart_phones1.csv'.")
