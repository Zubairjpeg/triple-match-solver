"""
Streamlit app: Triple Match 3D solver with template-matching CV detection.

Two pages:
  - Solve: upload screenshot -> detect tiles -> run DFS solver -> annotated image + move list
  - Templates: upload cropped tile images with labels, manage the template library

Notes on persistence:
  - Templates live in templates/ on disk. On Streamlit Cloud free tier, the disk
    is ephemeral across cold starts. The "Export ZIP" / "Import ZIP" buttons let
    you back up your library and reload it after a restart. For a permanent
    library, commit templates/*.png to the GitHub repo backing this app.
"""
from __future__ import annotations
import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from board import Tile, Board, GameState
from cv_detector import (
    load_templates,
    detect_board,
    annotate_image,
)
from solver import Solver


TEMPLATES_DIR = Path("templates")
TEMPLATES_DIR.mkdir(exist_ok=True)


st.set_page_config(page_title="Triple Match Solver", page_icon="🎯", layout="centered")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_to_bgr(pil_image: Image.Image) -> np.ndarray:
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Sidebar nav
# ---------------------------------------------------------------------------

page = st.sidebar.radio("Page", ["Solve", "Templates", "About"])
templates_now = load_templates(TEMPLATES_DIR)
st.sidebar.caption(f"Templates loaded: {len(templates_now)}")


# ---------------------------------------------------------------------------
# Page: Solve
# ---------------------------------------------------------------------------
if page == "Solve":
    st.title("Triple Match Solver")

    if not templates_now:
        st.warning("No templates yet. Go to the **Templates** page to upload one cropped image per tile type.")
        st.stop()

    with st.expander(f"Loaded {len(templates_now)} tile types"):
        st.write(", ".join(sorted(templates_now.keys())))

    uploaded = st.file_uploader(
        "Upload a screenshot of the board",
        type=["png", "jpg", "jpeg"],
        key="solve_upload",
    )

    if uploaded is None:
        st.info("Take a screenshot in-game and upload it here.")
        st.stop()

    pil_img = Image.open(uploaded).convert("RGB")
    bgr = pil_to_bgr(pil_img)

    with st.spinner("Detecting tiles..."):
        state, debug = detect_board(bgr, TEMPLATES_DIR)

    counts = state.board.type_counts()
    st.write(f"**Detected:** {len(state.board.tiles)} tiles across {len(counts)} types")

    bad_counts = {k: v for k, v in counts.items() if v % 3 != 0}
    if bad_counts:
        st.warning(
            "Some tile types don't have a multiple of 3 — detection probably missed or "
            "double-counted some tiles:\n\n" +
            ", ".join(f"`{k}`: {v}" for k, v in sorted(bad_counts.items()))
        )

    st.subheader("Detected board")
    annotated = annotate_image(bgr, debug)
    st.image(bgr_to_pil(annotated), width="stretch")

    if st.button("Solve"):
        with st.spinner("Solving..."):
            solver = Solver(state)
            solution = solver.solve()

        if solution is None:
            st.error(
                "No solution found. Likely causes: missed tiles in detection, or the "
                "current board is genuinely unsolvable (some tile type has < 3 copies)."
            )
        else:
            st.success(f"Solved in {len(solution)} moves")
            for i, move in enumerate(solution[:5], start=1):
                st.markdown(f"**{i}.** {move}")
            if len(solution) > 5:
                with st.expander(f"Show all {len(solution)} moves"):
                    for move in solution[5:]:
                        st.text(move)


# ---------------------------------------------------------------------------
# Page: Templates
# ---------------------------------------------------------------------------
elif page == "Templates":
    st.title("Manage tile templates")
    st.markdown(
        "Upload **one tightly-cropped image** per tile type. Crop in your phone's Photos app, "
        "save the crop, then upload here."
    )

    # --- Current templates ---
    existing = sorted(p.stem for p in TEMPLATES_DIR.glob("*.png"))
    if existing:
        st.subheader(f"Current library ({len(existing)})")
        cols_per_row = 4
        for row_start in range(0, len(existing), cols_per_row):
            row = existing[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, name in zip(cols, row):
                with col:
                    img = Image.open(TEMPLATES_DIR / f"{name}.png")
                    st.image(img, caption=name, width="stretch")
                    if st.button("Delete", key=f"del_{name}"):
                        (TEMPLATES_DIR / f"{name}.png").unlink(missing_ok=True)
                        st.rerun()
    else:
        st.info("No templates yet. Add your first one below.")

    # --- Add new ---
    st.markdown("---")
    st.subheader("Add new template")
    new_upload = st.file_uploader(
        "Cropped tile image",
        type=["png", "jpg", "jpeg"],
        key="new_template",
    )
    new_name_raw = st.text_input("Tile name (e.g. carrot, salmon, chicken_leg)")
    new_name = slugify(new_name_raw)

    if new_upload and new_name:
        preview = Image.open(new_upload).convert("RGB")
        st.image(preview, caption=f"Preview: {new_name}", width=150)
        if st.button("Save template"):
            out_path = TEMPLATES_DIR / f"{new_name}.png"
            preview.save(out_path, format="PNG")
            st.success(f"Saved {out_path.name}")
            st.rerun()

    # --- Backup / restore ---
    st.markdown("---")
    st.subheader("Backup / restore")
    st.caption(
        "Streamlit Cloud's free tier may wipe uploaded files on cold start. "
        "Download a ZIP of your library here and re-upload it later to restore."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if existing:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name in existing:
                    zf.write(TEMPLATES_DIR / f"{name}.png", arcname=f"{name}.png")
            st.download_button(
                "Download templates.zip",
                data=buf.getvalue(),
                file_name="templates.zip",
                mime="application/zip",
            )
    with col_b:
        restore = st.file_uploader("Restore from ZIP", type=["zip"], key="restore_zip")
        if restore is not None:
            with zipfile.ZipFile(restore) as zf:
                count = 0
                for name in zf.namelist():
                    if not name.lower().endswith(".png"):
                        continue
                    safe = Path(name).name  # strip dir traversal
                    with zf.open(name) as src:
                        (TEMPLATES_DIR / safe).write_bytes(src.read())
                    count += 1
            st.success(f"Restored {count} templates")
            st.rerun()


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
else:
    st.title("About")
    st.markdown(
        """
**Triple Match 3D solver.** Upload a screenshot, get an ordered move list.

**How it works:**
1. **Template matching** (OpenCV) finds each tile by comparing the screenshot
   against your saved tile crops.
2. **Grid clustering** snaps matches onto rows/columns.
3. **DFS solver** finds a clearing sequence using pruning (orphan detection,
   tray-overflow check, triplet priority).

**To get started:**
- Go to **Templates**, upload one crop per tile type (~10–15 total).
- Switch to **Solve** and upload a board screenshot.

**Tips:**
- Tighter crops (just the tile, no border) match better.
- If detection misses tiles, try uploading additional crops of the same type
  (e.g. a brighter version and a greyer/blocked version).
- Tile counts must each be multiples of 3 for the solver to find a full clearance.
"""
    )
