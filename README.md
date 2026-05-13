# Triple Match 3D Solver

Streamlit app that solves Triple Match 3D food-tile puzzles. Upload a screenshot → it detects the tiles via OpenCV template matching → runs a DFS solver → returns an ordered move list with an annotated image.

## What's in this repo

| File | Purpose |
|---|---|
| `board.py` | Tile / Board / GameState data structures |
| `solver.py` | DFS solver with orphan pruning + triplet-priority heuristic |
| `cv_detector.py` | OpenCV template matching → grid-snapped tile list |
| `app.py` | Streamlit UI (Solve + Templates pages) |
| `main.py` | CLI fallback (manual board input) |
| `requirements.txt` | streamlit, opencv-python-headless, numpy, Pillow |
| `templates/` | One PNG per tile type, filename = tile name |

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Deploy to Streamlit Cloud (free)

1. **Create a GitHub repo** with all the files in this directory. From iPhone: github.com → New → upload files.
2. Go to **share.streamlit.io** → sign in with GitHub.
3. Click **Create app** → pick your repo, branch `main`, main file `app.py`.
4. Click **Deploy**. ~2 minutes later you get a public URL like `https://<your-app>.streamlit.app`.
5. Bookmark that URL on your iPhone home screen.

No API keys needed.

## First-time setup: build your template library

The solver needs one cropped image per tile type. You only do this once.

1. Take a clear in-game screenshot where every tile type is visible.
2. On your iPhone: open the screenshot in **Photos** → tap **Edit** → **Crop** → tighten the crop around one tile → **Done** → **Save as new photo**.
3. Open the Streamlit app → **Templates** page.
4. Upload the crop, type the tile name (`carrot`, `salmon`, `chicken_leg`, etc.), click **Save**.
5. Repeat for every tile type (typically 10–15).
6. Click **Download templates.zip** as a backup.

When new stages introduce new tile types, repeat for the new ones.

## Streamlit Cloud free-tier storage caveat

Files in `templates/` persist while the app is warm but may be wiped on cold restart. Two ways to make templates permanent:

- **Easy:** keep `templates.zip` on your phone. After a cold start, go to Templates → Restore from ZIP.
- **Permanent:** commit your `templates/*.png` to the GitHub repo (via the GitHub mobile site: navigate to `templates/` → Add file → Upload files). They'll redeploy automatically.

## Troubleshooting

- **"No solution found"** — likely missed tiles. The warning above the board will flag any tile type with a non-multiple-of-3 count. Add more template variations (e.g., upload separate crops for bright vs greyed-out versions of the same tile).
- **Tiles detected as the wrong type** — your templates aren't distinctive enough. Re-crop tighter around the unique part of each tile.
- **App is slow** — Streamlit Cloud's free tier sleeps after ~15 min of inactivity. First request after a sleep takes ~30s to wake up.
