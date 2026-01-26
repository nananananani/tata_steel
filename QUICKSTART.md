# 🚀 Quick Start Guide - Tata Steel Ring Test

## Step 1: Install Python Dependencies

First, make sure you have Python 3.8+ installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install fastapi uvicorn opencv-python numpy pillow python-multipart requests
```

## Step 2: Test the Ring Detection (Standalone)

Test the ring detection algorithm without the API:

```bash
python test_ring.py
```

Choose option 1 to test with a synthetic image (auto-generated).

## Step 3: Start the API Server

```bash
python app.py
```

The server will start on http://localhost:8000

You should see:
```
============================================================
🏭 Tata Steel Rebar Testing API
============================================================
Starting server on http://localhost:8000
API Documentation: http://localhost:8000/docs
============================================================
```

## Step 4: Test the API

### Option A: Use the Interactive Docs

1. Open your browser to http://localhost:8000/docs
2. Try the `/api/ring-test` endpoint
3. Upload an image and select diameter
4. Click "Execute"

### Option B: Use the Test Script

In a new terminal:

```bash
python test_api.py
```

Choose option 1 to test health check, or option 4 to test ring detection.

### Option C: Use curl

```bash
curl -X POST "http://localhost:8000/api/ring-test" \
  -F "file=@your_image.jpg" \
  -F "diameter=12"
```

## Step 5: View Results

The API returns:
- **status**: PASS or FAIL
- **reason**: Explanation
- **level1**: Qualitative checks (4 tests)
- **level2**: Quantitative measurements
- **debug_image_url**: URL to view annotated image

Example:
```json
{
  "status": "PASS",
  "reason": "All tests passed. Thickness: 0.95mm",
  "level1": {
    "regions_visible": true,
    "ring_continuous": true,
    "concentric": true,
    "thickness_uniform": true
  },
  "level2": {
    "thickness_mm": 0.95,
    "within_standard": true
  },
  "debug_image_url": "/static/ring_debug_20260125_155122.jpg"
}
```

View the debug image at: http://localhost:8000/static/ring_debug_20260125_155122.jpg

## Common Commands

### Start Server
```bash
python app.py
```

### Test Ring Detection (Standalone)
```bash
python test_ring.py
```

### Test API
```bash
python test_api.py
```

### Test with Specific Image
```bash
python test_ring.py path/to/image.jpg 12
```

### Test API with Specific Image
```bash
python test_api.py test path/to/image.jpg 12
```

## Troubleshooting

### "pip not recognized"
Try: `python -m pip install -r requirements.txt`

### "python not recognized"
Try: `py -m pip install -r requirements.txt`

### Server won't start
- Check if port 8000 is already in use
- Try: `uvicorn app:app --port 8001`

### "Rings not detected"
- Ensure good lighting
- Image should be in focus
- Cross-section should be centered
- Try the synthetic test image first

## Next Steps

1. ✅ Test with synthetic images
2. ✅ Test with real rebar images
3. ✅ Integrate with your frontend
4. ✅ Coordinate with teammate on rib test

## File Structure

```
tata_steel/
├── app.py              # FastAPI server (START HERE)
├── ring_pipeline.py    # Your ring test logic
├── utils.py            # Shared utilities
├── test_ring.py        # Test ring detection
├── test_api.py         # Test API endpoints
├── requirements.txt    # Dependencies
├── README.md          # Full documentation
└── QUICKSTART.md      # This file
```

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Check the troubleshooting section
3. Contact the development team

---

**Happy Testing! 🎉**
