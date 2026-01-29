# Tata Steel | Automated Rebar Testing Suite 🏭

A premium, high-precision quality inspection system for TMT (Thermo-Mechanically Treated) bars, specifically designed for Tata Steel's engineering standards (IS 1786).

## 🌟 Overview

This suite utilizes advanced **Computer Vision** and **Signal Processing** algorithms—including **Otsu's Adaptive Thresholding**, **Probabilistic Hough Transform**, and **Structural Peak Detection**—to automate the inspection of rebar cross-sections (Ring Test) and longitudinal patterns (Rib Test) with 100% mathematical auditability.

## 📊 Modules

### 1. Ring Test (Cross-Section Analysis)
**Current Version: v4.1 (Variability Analysis)**
High-precision measurement of the Tempered Martensite (TM) ring morphology.
- **Level 1: Qualitative Check**
  - Dark & Light region separation logic.
  - Ring continuity & concentricity verification.
- **Level 2: Dimensional Check & Variability**
  - **Equivalent Area Method**: Calculates standard average thickness.
  - **Multi-Point Scanning**: Analyzes 360-degree thickness variability to report Min/Max spans.
  - **Target Window**: Automatic PASSED/FAILED decision based on diameter-specific standards.

### 2. Rib Test (Longitudinal Analysis)
**Current Version: v4.0 (Engineering Precision)**
Structural signal analysis system designed for industrial precision.
- **Localization**: **HSV Chromatic Isolation** to filter specifically for steel textual signatures and reject backgrounds.
- **Angle Detection**: **Probabilistic Hough Transform** for precise transverse rib angle measurement.
- **Metric Calculation**:
  - **Projected Rib Area ($A_R$)**: Uses the official IS 1786 formula: $1.33 \times L \times H \times \sin(\theta) / \text{Spacing}$.
  - **Signal Processing**: `scipy.signal` for robust peak detection (rib counting).

## 🎨 Premium Web Interface

The system features a modern "Industrial 4.0" dashboard:
- **Glassmorphic UI**: High-contrast dark mode optimized for factory lighting.
- **Mobile-First Design**: Fully responsive interface for real-time inspection via smartphone.
- **Real-Time Analysis**: <200ms processing time per image.

## 🚀 Installation & Setup

### 1. Requirements
- Python 3.9+
- Network access (for mobile usage)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Cloudinary credentials (optional, only needed for AI upscaling)
# Get free credentials at https://cloudinary.com
```

Required environment variables in `.env`:
- `CLOUDINARY_CLOUD_NAME`: Your Cloudinary cloud name
- `CLOUDINARY_API_KEY`: Your Cloudinary API key  
- `CLOUDINARY_API_SECRET`: Your Cloudinary API secret

> **Note**: Cloudinary is only required if you enable AI image upscaling. The system works without it.

### 4. Start the Suite
```bash
python run.py
```

### 📱 Mobile Access
The system automatically detects your local IP address.
1. Make sure your phone and PC are on the **same WiFi network**.
2. Run the server. It will display a specialized URL (e.g., `http://192.168.x.x:8001`).
3. Enter this URL in your phone's browser to upload images directly from the camera.

## 📂 Project Structure

```
tata_steel/
├── app.py              # FastAPI Backend (Endpoints for Ring & Rib)
├── ring_pipeline.py    # Ring Test Engine (Otsu + Morphology + Geometric)
├── rib_pipeline.py     # Rib Test Engine (HSV + Hough + Signal Processing)
├── run.py              # Server Entry Point (Auto-IP Detection)
├── static/             # Frontend Assets
│   ├── index.html      # Landing Page
│   ├── ring_test.html  # Ring Test Dashboard
│   ├── rib_test.html   # Rib Test Dashboard
│   └── styles.css      # Premium Design System
├── uploads/            # Temporary storage for analysis
└── requirements.txt    # Dependencies
```

## 🧠 Technology Stack
- **FastAPI**: High-performance async backend.
- **OpenCV**: Core image processing.
- **SciPy**: Advanced signal processing for rib detection.
- **TailwindCSS**: Utilitarian styling framework.

## 🤝 Contributors
- **Armaan Patel**: Lead Developer & AI Integration.

---
**Confidential Property of Tata Steel | Digital Quality Assurance Suite 2026**
