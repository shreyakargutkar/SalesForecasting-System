# Deployment Instructions - GitHub & Streamlit Community Cloud

Follow these steps to push your code to GitHub and host your live interactive sales portal on Streamlit Community Cloud.

## Step 1: Initialize Git and Push to GitHub

1. **Open your Terminal** and navigate to your project directory:
   ```bash
   cd /Users/smitakargutkar/Desktop/finalproject
   ```
2. **Initialize a local Git repository**:
   ```bash
   git init
   ```
3. **Create a `.gitignore` file** to prevent uploading unnecessary virtual environment or cache files:
   ```bash
   echo ".venv/" >> .gitignore
   echo "__pycache__/" >> .gitignore
   echo ".DS_Store" >> .gitignore
   ```
4. **Stage and commit your project files**:
   ```bash
   git add .
   git commit -m "Initial commit: sales analytics forecasting and operations portal"
   ```
5. **Create a new repository on GitHub**:
   - Go to [GitHub](https://github.com/) and log in.
   - Click the green **"New"** button to create a repository.
   - Name it `finalproject-sales-portal` (or any name you prefer).
   - Keep the repository **Public** (Streamlit Community Cloud is free for public repositories).
   - Do **NOT** initialize it with a README, `.gitignore`, or license.
   - Click **"Create repository"**.
6. **Link your local repository to GitHub and push**:
   - Copy the commands shown on GitHub under "or push an existing repository from the command line".
   - It will look like this:
     ```bash
     git branch -M main
     git remote add origin https://github.com/your-username/finalproject-sales-portal.git
     git push -u origin main
     ```
     *(Replace `your-username` with your actual GitHub username)*

---

## Step 2: Deploy on Streamlit Community Cloud

1. **Sign up or log in to Streamlit**:
   - Go to [Streamlit Community Cloud](https://share.streamlit.io/).
   - Click **"Sign in with GitHub"** to link your GitHub account.
2. **Launch a new deployment**:
   - Once logged in, click the **"New app"** button.
3. **Configure your deployment fields**:
   - **Repository**: Select `your-username/finalproject-sales-portal` from the dropdown list.
   - **Branch**: Set to `main`.
   - **Main file path**: Enter `app.py`.
   - **App URL**: You can optionally customize the subdomain (e.g. `superstore-operations-portal.streamlit.app`).
4. **Deploy**:
   - Click the blue **"Deploy!"** button.
   - Streamlit will automatically read your `requirements.txt` file, install all the required Python packages (such as `xgboost`, `prophet`, `statsmodels`), and spin up your application.
   - Within 2-3 minutes, your live link will be ready!
