# Tata Steel | Automated Rebar Testing Suite 🏭

A premium, AI-driven quality inspection system for TMT (Thermo-Mechanically Treated) bars, specifically designed for Tata Steel's high-precision standards.

## 🌟 Overview

This suite utilizes state-of-the-art computer vision models—including **YOLOv8**, **Segment Anything (SAM)**, and **Gabor Pattern Detection**—to automate the inspection of rebar cross-sections (Ring Test) and longitudinal patterns (Rib Test).

## 📊 Modules

### 1. Ring Test (Cross-Section Analysis)
High-precision measurement of the Tempered Martensite (TM) ring morphology.
- **Level 1: Qualitative Check**
  - Dark & Light region separation logic.
  - Ring continuity & concentricity verification.
  - Thickness uniformity assessment.
- **Level 2: Dimensional Check**
  - Millimeter-accurate thickness calculation.
  - Automatic PASSED/FAILED decision based on diameter-specific standards (8mm to 16mm).
  - Explicit Target Window (Min/Max range) display.

### 2. Rib Test (v3.0 High-Accuracy Engine)
**[NEW ARCHITECTURE]**
Hybrid Deep Learning & Periodic Signal Analysis system designed for industrial precision.
- **Localization**: YOLOv11 for intelligent rebar isolation and noise rejection.
- **Deep Segmentation**: Gabor Frequency-Domain Mapping (DeepLabV3+ style) for texture-based rib extraction.
- **Interval Assessment**: Signal periodicity analysis (DVNet-style) for sub-pixel inter-distance measurement.
- **Metrics Calculated**:
  - Number of ribs (Peak Detection)
  - Transverse Angle
  - Sub-pixel rib height
  - AR Value (Area Relative) calculation

## 🎨 Premium Web Interface

The system features a modern "Industrial 4.0" dashboard:
- **Glassmorphic UI**: Dark-themed, transparent panel design with vibrant accents.
- **Interactive Landing Page**: Seamless navigation between testing modules.
- **3-Column Dashboard**: 
  - **Left**: Live configuration & Intelligent Image Cropper.
  - **Center**: Real-time analytical results & Acceptance Criteria checkboxes.
  - **Right**: High-contrast Status Badges and "Spectral" Visual Detection Maps.

## 🚀 Installation & Setup

### 1. Requirements
- Python 3.9+
- CUDA-compatible GPU (Optional, for faster SAM/YOLO performance)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Suite
```bash
python run.py
```
Access the dashboard at: `http://localhost:8000`

## 📂 Project Structure

```
tata_steel/
├── app.py              # FastAPI Backend (Endpoints for Ring & Rib)
├── ring_pipeline.py    # Ring Test Engine (OpenCV + Geometric Logic)
├── rib_pipeline.py     # Rib Test Engine (YOLOv8 + SAM + Gabor Filters)
├── run.py              # Server Entry Point
├── static/             # Frontend Assets
│   ├── index.html      # Landing Page
│   ├── ring_test.html  # Ring Test Dashboard
│   ├── rib_test.html   # Rib Test Dashboard
│   ├── styles.css      # Premium Design System
│   ├── common.js       # Shared UI Logic (Cropper, Uploads)
│   ├── ring_test.js    # Ring Test Logic
│   └── rib_test.js     # Rib Test Logic
├── uploads/            # Temporary storage for analysis
└── requirements.txt    # AI & Backend 
```

## 🧠 AI Models Used
- **YOLOv8**: Object detection for rebar localization.
- **SAM (Segment Anything)**: Foundation model for sub-pixel boundary detection.
- **Gabor Filter Bank**: Mathematical pattern recognition for diagonal textures.

## 🤝 Contributors
- **Armaan Patel**: Lead Developer & AI Integration.

---
**Confidential Property of Tata Steel | Digital Quality Assurance Suite 2026**
