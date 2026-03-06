# SMART DEPTH VISION
## Real-Time Monocular Spatial Awareness Using YOLOv8 and Deep Depth Estimation

---

## Files in this project

| File | Purpose |
|---|---|
| `pipeline.py` | Core logic — YOLOv8 + MiDaS + 2D/3D classifier |
| `run_demo.py` | Runs full pipeline on all scenes, saves results |
| `test_pipeline.py` | Quick test on 2 images — run this FIRST |
| `fix_nyu_labels.py` | Fixes black NYU label images |
| `requirements.txt` | All dependencies |
| `install.bat` | One-click installer for Windows |

---

## How to run — Step by Step

### Step 1 — Install dependencies
Double-click `install.bat` OR run in terminal:
```
pip install -r requirements.txt
```

### Step 2 — Quick test first
```
python test_pipeline.py
```
Check that 2 result images appear in the results/test folder.
If they look correct (RGB + Depth side by side with boxes), proceed.

### Step 3 — Run full demo
```
python run_demo.py
```
Results saved to: `C:\Users\Ankush\Desktop\Btech\Capstone\results\`

### Step 4 — Fix NYU black labels (separate)
Update path in fix_nyu_labels.py then:
```
python fix_nyu_labels.py
```

---

## How the 2D/3D classification works

```
RGB Image
    │
    ├──► YOLOv8 ──────────► Detected objects with bounding boxes
    │
    └──► MiDaS ───────────► Depth map (how far each pixel is)
                                │
                                └──► For each bounding box:
                                        Compute std deviation of depth values
                                        
                                        HIGH std dev = object has varying depth
                                                     = real 3D object in world
                                        
                                        LOW std dev  = object has flat uniform depth
                                                     = 2D representation (poster/screen)
```

---

## Output format

Each result image shows:
- **Left panel**: Original RGB with YOLOv8 boxes + 2D/3D label
  - GREEN box = 3D object (real world)
  - RED box = 2D object (flat/screen/poster)
- **Middle panel**: MiDaS depth colormap
- **Right panel**: Ground truth depth (from dataset)

---

## If something doesn't work

**Models not downloading?**
- Check internet connection (MiDaS downloads from PyTorch Hub first time)

**CUDA not available?**
- Run: `python -c "import torch; print(torch.cuda.is_available())"`
- Pipeline still works on CPU, just slower

**Path errors?**
- Update `DATASET_PATH` in `run_demo.py` to your exact path
