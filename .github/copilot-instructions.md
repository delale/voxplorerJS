# Voxplorer Copilot Instructions

## Project Overview

Voxplorer is a Dash web application for interactive voice data visualization and analysis. Users upload pre-computed features (CSV/TSV/XLSX) or audio files to extract acoustic features (MFCCs or speaker embeddings), apply dimensionality reduction (PCA, UMAP, t-SNE, MDS), and explore results through interactive 2D/3D plots synced with filterable tables.

**Tech Stack**: Python 3.12+, Dash/Plotly (frontend), scikit-learn/UMAP/librosa/speechbrain (backend)

## Architecture

### Data Flow

1. **Input Layer** (`pages/layouts/visualiser/`):
   - `table_upload/`: Parse CSV/TSV/XLSX → Polars DataFrame
   - `audio_upload/`: Upload WAV/MP3/FLAC → feature extraction

2. **Processing Layer** (`lib/`):
   - `feature_extraction.py`: `FeatureExtractor` class extracts MFCCs (with delta/delta-delta options) or speaker embeddings (speechbrain pre-trained models)
   - `data_loader.py`: Utility functions for table parsing and audio processing
   - `dimensionality_reduction.py`: Four algorithms (PCA, UMAP, t-SNE, MDS) with supervised/unsupervised modes
   - `plotting.py`: Plotly scatter plots (2D/3D) with selection highlighting

3. **State Management** (`pages/visualiser.py`):
   - Dash `dcc.Store` components manage session data (raw table, metadata, reduced dims, selections, logs)
   - Callbacks sync selections between plot and table view
   - Processing logs (JSON) capture all parameters for reproducibility

### Key Patterns

- **Modular callbacks**: Each layout component (`table_preview`, `plot_layout`, `dimensionality_reduction_opts`) has isolated callback logic
- **Data serialization**: All data in stores uses dict/list (JSON-compatible), converted to Polars/pandas only when needed
- **Error handling**: Functions return tuple `(result, alert_component)` for user feedback
- **Feature labels**: Automatically generated (`c1`, `c2` for MFCCs; `d1`, `d2` for delta; `X1`, `X2` for embeddings)

## Developer Workflows

### Running the App

```bash
# Using uv (recommended)
uv run app.py

# Using pip
pip install -r requirements.txt
python3 app.py
```

Access at `http://127.0.0.1:8050/`

### Testing

```bash
# Run all tests
pytest test/

# Run specific test file
pytest test/test_feature_extraction.py

# Run specific test class or function
pytest test/test_feature_extraction.py::TestFeatureExtractor::test_method_name

# Run with coverage
pytest test/ --cov=lib --cov=pages

# Run with verbose output
pytest test/ -v
```

Test data is in `test/data/` (e.g., `sp01_sample1.wav` for audio tests). Tests use `unittest.TestCase` with setUp/tearDown patterns. Always import from `lib/` modules using relative paths.

### Linting & Code Quality

```bash
# Lint Python files (using ruff)
uv run ruff check .

# Format Python files (auto-fix where possible)
uv run ruff format .

# Check specific directory
uv run ruff check lib/
```

**Ruff is configured via pyproject.toml** (see dev dependencies). Both linting and formatting are available but not enforced via CI yet.

### Dependencies Management

- **Tool**: `uv` (https://docs.astral.sh/uv/)
- **Config**: `pyproject.toml` specifies all dependencies including dev tools (ruff, ipython)
- **Update after modifying pyproject.toml**: `uv export --no-hashes --format requirements-txt > requirements.txt`
- **Add a new dependency**: Edit `pyproject.toml` dependencies or dev dependencies, then regenerate requirements.txt

## Code Patterns & Architecture

### Dash Callbacks
- Each layout component file has isolated callbacks registered with `@callback` decorator
- Use `callback_context` (`ctx`) to identify which input triggered the callback via `ctx.triggered_id`
- Multi-output callbacks use list syntax: `Output()` returns a list matching output order
- All callback functions should prevent unnecessary recalculation with guards (`PreventUpdate`, check `ctx.triggered_id`)
- State components read current value without triggering callback; use for optional UI state (e.g., visibility)

### Error Handling & User Feedback
- Functions that may fail should return tuple `(result, alert_component)`
  - `result`: computed data or `None` if error
  - `alert_component`: `dbc.Alert` with error message or `None` if success
- Example pattern:
  ```python
  def process_upload(filename, contents):
      try:
          # do work
          return data, None
      except Exception as e:
          alert = dbc.Alert(str(e), color="danger")
          return None, alert
  ```

### Page Registration
- Pages are registered using `dash.register_page(__name__, path="/route-name")`
- Entry point: `app.py` sets up navbar and page routing with `use_pages=True`
- Home page: `pages/home.py`
- Add new pages in `pages/` directory with `register_page()` call
- Incomplete: `recogniser.py` page exists but is not yet wired to navbar (see todo.md)

### Data Serialization & Stores
- `dcc.Store` uses `storage_type="memory"` for session-only data (lost on refresh)
- Data in stores must be JSON-serializable (dicts/lists, not DataFrames/numpy arrays)
- Convert Polars/pandas to dicts before storing: `.to_dicts()` or `.to_dict('records')`
- Always convert back to Polars/pandas in callback when needed for computation
- Stores act as "global session state"; multiple callbacks can read/write same store

## Critical Conventions

### Audio Processing (`feature_extraction.py`)

- **Input**: Base64-encoded file bytes (`data:audio/wav;base64,XXX...`)
- **Metadata extraction**: Parse filename by split character (default `_`) to extract variables (e.g., `speaker_emotion_take.wav`)
- **Summarization**: When `summarise=True`, MFCC frames averaged to single vector (mean + std per feature)
- **Resampling**: Speaker embedding model requires 16 kHz; `torchaudio.functional.resample()` handles conversion
- **Pre-trained models**: Cached in `.pretrained_spkrec_models/` directory

### Table Handling

- **Library**: Polars (preferred) or pandas for compatibility
- **Indexing**: Always add `row_index` column after loading (`with_row_index("row_index")`)
- **Serialization**: Convert to list of dicts (`to_dicts()`) for JSON store
- **Filtering**: Table filtering doesn't affect selections; use `derived_virtual_data` for filtered view state

### Dimensionality Reduction

- Output always has columns named `DIM1`, `DIM2`, `DIM3`
- All algorithms handle sklearn-compatible APIs
- UMAP supports supervised mode (provide `y` parameter); others are unsupervised only
- Explained variance/singular values returned for PCA only

### Plotting & Selection

- **2D plots**: Call `scatter_2d(data, x, y, ...)` with color/symbol grouping
- **3D plots**: Call `scatter_3d(data, x, y, z, ...)` 
- **Selection sync**: Plot selections (`customdata[0]`) and table selections both store row indices
- **Highlight style**: Selected points remain visible; unselected faded (opacity ~0.2)

## Debugging Workflows

### Common Issues & Diagnostics

**Upload callbacks not firing:**
- Check `ctx.triggered_id` in callback—ensure you're checking the correct trigger
- Verify `dcc.Upload` component IDs match callback inputs
- Test with simple print statements; use browser console for JavaScript errors

**Data not appearing in table/plot:**
- Verify data is being stored correctly: inspect store contents in browser DevTools (F12 → Application → Cookies/Storage)
- Check if row_index column was added after loading table
- Ensure serialization to/from dict format is correct (no NaN/inf values which break JSON)
- Test data shape and value ranges with `print()` in callbacks

**Selections not syncing between plot and table:**
- Plot selections use `customdata[0]` to store row indices; table selections use `derived_virtual_data`
- Both stores are updated independently; check `selected-observations` store
- If selections don't highlight, check CSS opacity settings in `assets/styles.css`

**Dimensionality reduction fails silently:**
- Check algorithm parameters in `dimensionality_reduction_opts.py`—UMAP is sensitive to n_neighbors
- Verify input data shape: should be (n_samples, n_features)
- Test with small dataset first; algorithms may be slow on large data
- Check logs in processing-logs store for error details

**Audio feature extraction hangs:**
- First-run downloads speechbrain model; this can take time
- Check `.pretrained_spkrec_models/` directory exists
- If using speaker embeddings, verify audio is valid WAV/MP3/FLAC

### Testing Your Changes

1. **Unit tests**: Write tests in `test/` following existing `unittest.TestCase` pattern
2. **Manual testing**: Use browser DevTools (F12) to:
   - Check Network tab for callback requests
   - Inspect Application storage for dcc.Store contents
   - Check Console for JavaScript errors from Plotly
3. **Integration**: Upload test data from `test/data/` to verify end-to-end flow

## Important File Locations

- **App entry point**: `app.py` (Dash app initialization, navbar, page routing, exit callback)
- **Primary page logic**: `pages/visualiser.py` (main callbacks, data promotion across stores)
- **Layout components**: `pages/layouts/visualiser/`
  - `table_preview.py`: Table display + filtering + selection sync
  - `plot_layout.py`: 2D/3D plot rendering + selection sync
  - `dimensionality_reduction_opts.py`: Algorithm selection and parameters
  - `audio_upload/audio_upload.py`: Audio file upload trigger
  - `audio_upload/feature_extraction_opts.py`: Feature extraction method options
  - `table_upload/table_upload.py`: Table file upload trigger
  - `table_upload/metaconfig.py`: Metadata variable mapping UI
- **Backend API**: `lib/`
  - `feature_extraction.py`: `FeatureExtractor` class (MFCCs, speaker embeddings)
  - `dimensionality_reduction.py`: PCA, UMAP, t-SNE, MDS implementations
  - `plotting.py`: Plotly scatter_2d/scatter_3d functions
  - `data_loader.py`: Table parsing (CSV/TSV/XLSX) and audio decoding utilities
- **Tests**: `test/` with test data in `test/data/`
- **Styles**: `assets/` (local CSS only; no external CDN dependencies)

## Common Tasks

**Add a new dimensionality reduction algorithm**:
1. Implement function in `lib/dimensionality_reduction.py` (signature: `func(X, **kwargs)`)
2. Update `dimensionality_reduction_opts.py` to add radio button option
3. Update visualiser.py callback to call new algorithm
4. Add tests in `test/test_dimensionality_reduction.py`

**Add a new feature extraction method**:
1. Add static method to `FeatureExtractor` class
2. Update `FeatureExtractor.process_files()` to invoke method
3. Update `FeatureExtractor.add_feature_labels()` for label generation
4. Update `audio_upload.py` UI to expose new feature options

**Debug table/audio uploads**:
- Check `parse_table_contents()` and `parse_audio_contents()` in `data_loader.py`
- Verify base64 encoding/decoding in feature extraction
- Log returns from callbacks (always check `ctx.triggered_id`)

## Offline Mode

The application is designed to work completely offline after initial setup. Key considerations:

### Styling
- **No CDN dependencies**: Uses local `assets/bootstrap_flatly_local.css` instead of `dbc.themes.FLATLY` (which would fetch from CDN)
- **Custom assets**: Additional styles in `assets/styles.css` and `assets/custom_classes.css`
- App is styled using Dash Bootstrap Components (dbc) components with local CSS only

### Pre-trained Models
- **Speaker embeddings**: First run downloads `speechbrain/spkrec-ecapa-voxceleb` model and caches in `.pretrained_spkrec_models/` directory
- **Offline usage**: After first download, can disconnect from internet and use cached models
- See `FeatureExtractor.speaker_embeddings()` in `lib/feature_extraction.py` for caching logic

### Data Privacy
- All processing happens locally in the Dash/Flask app—no data sent to external services
- Tables and audio files never leave the user's machine
- Processing logs stored locally

## Notes for Future Development

- Recogniser page is incomplete (see `todo.md`)
- Windows support may require modifications to app startup (currently optimized for macOS/Linux)
- Consider pre-packaging the speechbrain model to eliminate first-run download dependency
