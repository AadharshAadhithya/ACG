# ACG 

This is a work in progress. 

## Getting Started

### 1. **Clone the repository**
```bash
git clone https://github.com/AadharshAadhithya/ACG.git
cd ACG```

### **2. nstall required dependencies**
```bash
pip install -r requirements.txt
```

Ensure you have correct `.env` file. and source it `source .env` . Now run the setup script `./setup.sh`


This will:

-  Load the environment variables.
- Pull the data from S3 using DVC.


---

## Adding a New Folder (e.g., `pre_2`) to S3

### 1. **Add a new folder to DVC**
```bash
dvc add data/pre_2
```

### 2. **Commit changes to Git**
```bash
git add data/pre_2.dvc .gitignore
git commit -m "Add pre_2 data to DVC"
```

### 3. **Push the data to S3**
```bash
dvc push
```

### 4. **Push changes to Git**
```bash
git push
```
```
