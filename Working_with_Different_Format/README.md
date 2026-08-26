# Working with Different Data Formats

Welcome to the **Working with Different Data Formats** module! In the real world, data doesn't always come beautifully packaged in a perfectly clean CSV file. Data Scientists must be highly versatile, constantly extracting data from web APIs, relational databases, text files, and complex nested JSON structures.

This directory focuses on the practical Pandas skills required to confidently parse, extract, and convert data from multiple different file architectures into clean, model-ready DataFrames.

## Overview

Machine learning models fundamentally require flat, 2D arrays (rows and columns) of numbers. However, real-world data is often stored in:
1. **JSON (JavaScript Object Notation):** The universal standard for web APIs. JSON is highly nested (dictionaries inside lists inside dictionaries) and requires careful flattening.
2. **SQL Databases:** Massive relational databases requiring querying before extraction.
3. **CSV/TSV:** Comma or Tab-Separated Values, which frequently suffer from bad lines, incorrect encodings, or missing headers.

## Contents & Findings

### 1. [`Working with CSV Files.ipynb`](Working%20with%20CSV%20Files.ipynb)
**Goal:** Master the nuances of the `pd.read_csv()` engine.
* **Handling Bad Data:** Learning how to skip corrupted lines (`on_bad_lines='skip'`), handle files with no headers, and specify exact index columns.
* **Encoding Issues:** Solving the dreaded `UnicodeDecodeError` by explicitly specifying correct text encodings (like `ISO-8859-1` or `latin-1`).
* **Chunking:** How to load massive, gigabyte-sized CSV files in small, manageable batches (`chunksize`) to prevent your system's RAM from crashing.

### 2. [`Working with JSON - SQL.ipynb`](Working%20with%20JSON%20-%20SQL.ipynb)
**Goal:** Extract data directly from web APIs (JSON) and relational databases (SQL).
* **JSON Parsing:** Using `pd.read_json()` to seamlessly convert nested JSON data into flat, 2D Pandas DataFrames. 
* **SQL Integration:** We explore how to connect Pandas directly to a database engine (like SQLite) using `sqlalchemy`. We use `pd.read_sql()` to execute raw SQL queries (like `SELECT * FROM table`) and instantly dump the results directly into a working DataFrame.

## Getting Started
Launch the notebooks to learn how to source and clean data from anywhere on the web:
```bash
jupyter notebook "Working with CSV Files.ipynb"
```
