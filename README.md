# Tata Steel Rebar Testing API

Automated quality inspection system for TMT (Thermo-Mechanically Treated) bars using computer vision.

## 🎯 Features

### Ring Test (Implemented)
- **Level 1: Qualitative Analysis**
  - ✅ Dark & Light regions visibility check
  - ✅ Ring continuity verification
  - ✅ Concentricity validation
  - ✅ Thickness uniformity assessment

- **Level 2: Quantitative Analysis**
  - ✅ Precise thickness measurement
  - ✅ Dimensional analysis
  - ✅ Standard compliance validation
  - ✅ Visual debugging with annotated images

### Rib Test (To be implemented by teammate)
- Length measurement
- Angle calculation
- Height measurement
- Inter-distance analysis

## 📋 Requirements

- Python 3.8+
- OpenCV 4.9+
- FastAPI 0.109+
- NumPy 1.26+

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/nananananani/tata_steel.git
cd tata_steel
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install fastapi uvicorn opencv-python numpy pillow python-multipart
```

## 💻 Usage

### Starting the Server

```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### 1. Ring Test
```bash
POST /api/ring-test
```

**Parameters:**
- `file`: Image file (multipart/form-data)
- `diameter`: TMT bar diameter in mm (8, 10, 12, or 16)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/ring-test" \
  -F "file=@rebar_image.jpg" \
  -F "diameter=12"
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/api/ring-test"
files = {"file": open("rebar_image.jpg", "rb")}
data = {"diameter": 12}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Response:**
```json
{
  "status": "PASS",
  "reason": "All tests passed. Thickness: 0.95mm Within standard range (0.84-1.2mm)",
  "level1": {
    "regions_visible": true,
    "ring_continuous": true,
    "concentric": true,
    "thickness_uniform": true,
    "details": {
      "contrast_difference": 45.2,
      "ring_coverage_percent": 85.3,
      "center_distance_px": 3.5,
      "thickness_cv_percent": 8.2
    }
  },
  "level2": {
    "thickness_mm": 0.95,
    "within_standard": true,
    "standard_range": {
      "min": 0.84,
      "max": 1.2
    },
    "standard_message": "Within standard range (0.84-1.2mm)",
    "dimensions": {
      "inner_radius_px": 120.5,
      "outer_radius_px": 145.3,
      "thickness_px": 24.8,
      "thickness_mm": 0.95
    }
  },
  "debug_image_url": "/static/ring_debug_20260125_155122.jpg",
  "timestamp": "2026-01-25T15:51:22.123456"
}
```

#### 2. Get Thickness Standards
```bash
GET /api/standards
```

**Response:**
```json
{
  "standards": [
    {"diameter_mm": 8, "min_thickness_mm": 0.56, "max_thickness_mm": 0.8},
    {"diameter_mm": 10, "min_thickness_mm": 0.7, "max_thickness_mm": 1.0},
    {"diameter_mm": 12, "min_thickness_mm": 0.84, "max_thickness_mm": 1.2},
    {"diameter_mm": 16, "min_thickness_mm": 1.12, "max_thickness_mm": 1.6}
  ],
  "unit": "millimeters"
}
```

#### 3. Get Standard for Specific Diameter
```bash
GET /api/standards/{diameter}
```

#### 4. Health Check
```bash
GET /health
```

### Using the Ring Test Directly (Python)

```python
import cv2
from ring_pipeline import run_ring_test

# Load image
image = cv2.imread("rebar_cross_section.jpg")

# Run test
results = run_ring_test(image, diameter_mm=12)

# Check results
print(f"Status: {results['status']}")
print(f"Reason: {results['reason']}")

if results['level1']:
    print("\nLevel 1 Results:")
    print(f"  Regions Visible: {results['level1']['regions_visible']}")
    print(f"  Ring Continuous: {results['level1']['ring_continuous']}")
    print(f"  Concentric: {results['level1']['concentric']}")
    print(f"  Thickness Uniform: {results['level1']['thickness_uniform']}")

if results['level2']:
    print("\nLevel 2 Results:")
    print(f"  Thickness: {results['level2']['thickness_mm']:.2f}mm")
    print(f"  Within Standard: {results['level2']['within_standard']}")

# View debug image
debug_img = cv2.imread(results['debug_image_path'])
cv2.imshow("Debug", debug_img)
cv2.waitKey(0)
```

## 🔬 Technical Details

### Ring Detection Algorithm

1. **Preprocessing**
   - Grayscale conversion
   - Gaussian blur (noise reduction)
   - CLAHE (contrast enhancement)
   - Bilateral filtering (edge preservation)

2. **Circle Detection**
   - HoughCircles algorithm for outer ring (dark region)
   - Inverted HoughCircles for inner ring (light region)
   - Adaptive parameter tuning based on image size

3. **Validation**
   - Size validation
   - Concentricity check
   - Thickness reasonableness

4. **Analysis**
   - Contrast measurement
   - Continuity assessment
   - Uniformity calculation (coefficient of variation)
   - Dimensional measurements

### Recommended Image Guidelines

For best results:
- **Resolution**: Minimum 640x640 pixels
- **Lighting**: Uniform, diffused lighting
- **Focus**: Sharp focus on cross-section
- **Angle**: Perpendicular to cross-section
- **Background**: Dark, non-reflective background
- **Format**: JPEG or PNG

### HoughCircles Parameters

The algorithm uses adaptive parameters based on image size:

```python
# Outer ring detection
cv2.HoughCircles(
    preprocessed,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=image_width * 0.3,
    param1=50,
    param2=30,
    minRadius=min_dim * 0.15,
    maxRadius=min_dim * 0.45
)

# Inner ring detection
cv2.HoughCircles(
    inverted,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=image_width * 0.15,
    param1=50,
    param2=35,
    minRadius=min_dim * 0.05,
    maxRadius=min_dim * 0.35
)
```

## 📊 Thickness Standards

| Diameter (mm) | Min Thickness (mm) | Max Thickness (mm) |
|---------------|-------------------|-------------------|
| 8             | 0.56              | 0.8               |
| 10            | 0.7               | 1.0               |
| 12            | 0.84              | 1.2               |
| 16            | 1.12              | 1.6               |

## 🐛 Troubleshooting

### "Rings not detected"
- Ensure good lighting
- Check image focus
- Verify cross-section is centered
- Try different preprocessing parameters

### "Non-concentric rings"
- Verify cutting quality
- Check if image shows true cross-section
- May indicate actual manufacturing defect

### "Image quality issues"
- Increase resolution
- Improve lighting
- Reduce blur
- Adjust exposure

## 📁 Project Structure

```
tata_steel/
├── app.py              # FastAPI application
├── ring_pipeline.py    # Ring test implementation
├── rib_pipeline.py     # Rib test (teammate's work)
├── utils.py            # Shared utilities
├── requirements.txt    # Dependencies
├── static/             # Debug images
├── uploads/            # Temporary uploads
└── README.md          # This file
```

## 🤝 Team Responsibilities

- **Ring Test**: Your implementation (this repository)
- **Rib Test**: Teammate's implementation
- **Shared**: `app.py`, `utils.py`

## 📝 TODO

- [ ] Add batch processing endpoint
- [ ] Implement image preprocessing options
- [ ] Add parameter tuning interface
- [ ] Create test suite with sample images
- [ ] Add logging and monitoring
- [ ] Implement caching for repeated tests
- [ ] Add authentication for production

## 📄 License

Internal project for Tata Steel

## 👥 Contributors

- Ring Test Implementation: Armaan Patel
- Rib Test Implementation: [Teammate Name]

---

**For support or questions, contact the development team.**
