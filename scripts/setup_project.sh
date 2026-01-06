#!/bin/bash

# SMS Spam Detector - Automated Setup Script
# This script sets up the complete project from scratch

set -e  # Exit on error

PROJECT_NAME="${1:-sms-spam-detector}"

echo "=================================="
echo "SMS SPAM DETECTOR - PROJECT SETUP"
echo "=================================="
echo ""
echo "Setting up project: $PROJECT_NAME"
echo ""

# Create project directory
if [ -d "$PROJECT_NAME" ]; then
    echo "⚠️  Directory $PROJECT_NAME already exists!"
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Create directory structure
echo "Step 1/7: Creating directory structure..."
mkdir -p "$PROJECT_NAME"/{data/{raw,processed,feedback},src/{models,features,training,api,monitoring},tests,scripts,docs,docker,models,logs}

# Create .gitkeep files
touch "$PROJECT_NAME"/data/raw/.gitkeep
touch "$PROJECT_NAME"/data/processed/.gitkeep
touch "$PROJECT_NAME"/data/feedback/.gitkeep
touch "$PROJECT_NAME"/models/.gitkeep

echo "✅ Directory structure created"

# Step 2: Create Python virtual environment
echo ""
echo "Step 2/7: Creating Python virtual environment..."
cd "$PROJECT_NAME"

if command -v python3 &> /dev/null; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
else
    echo "❌ Python3 not found. Please install Python 3.10 or higher."
    exit 1
fi

# Step 3: Install dependencies
echo ""
echo "Step 3/7: Installing Python dependencies..."
echo "This may take 5-10 minutes..."

if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "⚠️  requirements.txt not found. Skipping..."
fi

# Step 4: Setup environment variables
echo ""
echo "Step 4/7: Setting up environment variables..."
if [ -f ".env.example" ]; then
    cp .env.example .env
    echo "✅ .env file created from .env.example"
    echo "   Edit .env to customize configuration"
else
    echo "⚠️  .env.example not found. Creating basic .env..."
    cat > .env << EOF
MODEL_TYPE=smart_router
API_PORT=8000
DEBUG=false
EOF
fi

# Step 5: Download datasets
echo ""
echo "Step 5/7: Downloading and preparing datasets..."
echo "This will download ~15,000 SMS messages..."

if [ -f "data/download_datasets.py" ]; then
    python data/download_datasets.py
    echo "✅ Datasets prepared"
else
    echo "⚠️  Dataset downloader not found. You'll need to run it manually:"
    echo "   python data/download_datasets.py"
fi

# Step 6: Git initialization
echo ""
echo "Step 6/7: Initializing Git repository..."
if command -v git &> /dev/null; then
    if [ ! -d ".git" ]; then
        git init
        git add .
        git commit -m "Initial commit: SMS Spam Detector project"
        echo "✅ Git repository initialized"
    else
        echo "ℹ️  Git repository already exists"
    fi
else
    echo "⚠️  Git not found. Skipping..."
fi

# Step 7: Summary and next steps
echo ""
echo "=================================="
echo "✅ SETUP COMPLETE!"
echo "=================================="
echo ""
echo "Project: $PROJECT_NAME"
echo "Location: $(pwd)"
echo ""
echo "📁 Directory Structure:"
echo "   ├── data/           # Datasets and feedback"
echo "   ├── src/            # Source code"
echo "   ├── models/         # Trained models"
echo "   ├── tests/          # Unit tests"
echo "   ├── docs/           # Documentation"
echo "   └── docker/         # Docker configuration"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. Activate virtual environment (if not already active):"
if [ -f "venv/bin/activate" ]; then
    echo "   source venv/bin/activate"
elif [ -f "venv/Scripts/activate" ]; then
    echo "   source venv/Scripts/activate"
fi
echo ""
echo "2. Train models (20-30 minutes):"
echo "   python src/training/train_all_models.py"
echo ""
echo "3. Start API server:"
echo "   uvicorn src.api.main:app --reload"
echo ""
echo "4. Test the API:"
echo "   curl -X POST http://localhost:8000/classify \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"message\": \"URGENT! Click bit.ly/verify\"}'"
echo ""
echo "5. View API documentation:"
echo "   http://localhost:8000/docs"
echo ""
echo "📚 Documentation:"
echo "   - README.md           # Project overview"
echo "   - docs/DATASET.md     # Dataset information"
echo "   - docs/MODELS.md      # Model comparison"
echo "   - docs/API.md         # API reference"
echo ""
echo "🐳 Docker Deployment:"
echo "   docker-compose up -d"
echo ""
echo "Need help? Check the README or documentation!"
echo "=================================="
