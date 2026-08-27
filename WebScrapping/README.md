# Web Scraping for Machine Learning

Welcome to the **Web Scraping** module! In Data Science and Machine Learning, the data you need won't always be neatly packaged in a Kaggle CSV file or provided via a clean JSON API. Often, the most valuable and unique data is locked away inside the HTML structure of live websites.

This directory focuses on the highly practical skill of **Web Scraping**: programmatically requesting web pages and extracting raw, unstructured HTML data into clean, structured Pandas DataFrames.

## Overview

Web scraping bridges the gap between raw internet data and machine learning pipelines. We primarily use two core Python libraries to accomplish this:

1. **`requests`:** Used to act as a web browser. It sends HTTP `GET` requests to a website URL and downloads the raw HTML source code.
2. **`BeautifulSoup` (bs4):** An HTML parser. It takes the giant block of raw text returned by `requests` and allows us to search through it using HTML tags (like `<h1>`, `<div>`, `<a>`) and CSS classes to extract exactly the text or numbers we need.

## Contents & Findings

### [`WebScraping -  GitHub.ipynb`](WebScraping%20-%20%20GitHub.ipynb)
**Goal:** Build a custom web scraper to extract live data directly from a target website (e.g., GitHub) and format it for statistical analysis.

* **The Process:**
  1. **Fetching the Page:** We use `requests.get(url)` to download the HTML content of the target page. We verify the connection by checking if the HTTP status code is `200` (Success).
  2. **Parsing the HTML:** We pass the raw text into `BeautifulSoup(html, 'html.parser')` to create a cleanly formatted, mathematically navigable tree of the website's DOM structure.
  3. **Targeted Extraction:** By inspecting the website's source code in the browser, we identify the exact HTML tags and class names holding our target data. We then use `.find()` or `.find_all()` to programmatically loop through and extract that information.
  4. **Structuring the Data:** Once we have extracted lists of raw strings, we clean them and load them directly into a Pandas `DataFrame` so they can be seamlessly integrated into a standard ML preprocessing pipeline.

## Getting Started
Launch the notebook to learn how to harvest your own custom datasets directly from the internet:
```bash
jupyter notebook "WebScraping -  GitHub.ipynb"
```
