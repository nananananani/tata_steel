# 🚀 Quick Start - Tata Steel Rebar Analysis

Follow these steps to get the Tata Steel Rebar Analysis Suite up and running in minutes.

## 1. Environment Setup

Make sure you have Python 3.9+ installed.

```bash
# Install all required AI and Web dependencies
pip install -r requirements.txt
```

*Note: The first time you run the Rib Test, it will automatically download the YOLOv8 weight files (~6MB).*

## 2. Launching the Suite

Start the integrated web server using the runner script:

```bash
python run.py
```

- **Dashboard:** [http://localhost:8000](http://localhost:8000)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

## 3. Running Your First Test

### Ring Test Analysis
1. Navigate to the **Ring Test** page from the dashboard.
2. Select your rebar diameter (e.g., 12mm).
3. Upload a high-resolution photo of the **cross-section**.
4. Use the **Crop Tool** to center the rebar cross-section.
5. Click **Analyze Ring Test**.

### Rib Test Analysis
1. Navigate to the **Rib Test** page.
2. Upload a **side-view** photo of the rebar.
3. Use the **Crop Tool** to isolate a segment with at least 3-4 distinct transverse ribs.
4. Click **Analyze Rib Test**.

## 4. Understanding Results

- **PASS/FAIL**: Based on official Tata Steel tolerance standards.
- **AR Value**: Only calculated for the Rib Test; target is typically > 0.040.
- **TM Ring Thickness**: Only calculated for the Ring Test; compared against diameter-specific targets.

## 📁 Key Files

- `app.py`: The "Brain" (FastAPI Backend).
- `static/`: The "Face" (HTML/CSS/JS Dashboards).
- `ring_pipeline.py`: Ring extraction logic.
- `rib_pipeline.py`: AI-driven rib segmentation.

---
**Happy Inspecting! 🎉**
