# TODO.md

Future development tasks and feature requests for the NIR Spectra Analyzer.

## Spectrum View Tab

### 🔄 Plot Type Toggle
**Priority: High**

Add ability to toggle between different plot types directly in the Spectrum View tab:
- Raw spectra (current default)
- 1st derivative
- 2nd derivative

**Rationale**: Second derivative often reveals more detail and is commonly needed for quick inspection before preprocessing. Users shouldn't need to go to Preprocessing tab just to view derivatives.

**Implementation Notes**:
- Add radio buttons or dropdown in the display controls
- Apply derivative calculation on-the-fly for visualization only
- Does not modify the underlying data (preprocessing tab still needed for saving processed data)
- Consider caching derivative calculations for performance with many files

---

### 🎯 Individual Spectrum Selection & Removal
**Priority: High**

Improve workflow for identifying and removing bad/outlier spectra:

**Current Issue**: Difficult to remove individual bad spectra from a batch of files. Zooming helps identify issues, but there's no easy way to focus on or remove specific spectra.

**Proposed Features**:
1. **Click-to-focus**: Click on a spectrum in the legend or plot to highlight it
2. **Solo mode**: Button to show only the selected spectrum (hide all others temporarily)
3. **Remove button**: After selecting a spectrum, option to remove it from the current session
4. **Outlier highlighting**: Visual indicators for spectra that are statistical outliers

**Implementation Notes**:
- Use Plotly click events to capture spectrum selection
- Add "Remove selected" button that updates session state
- Consider adding undo functionality for accidental removals
- May want to keep removed spectra in a separate list for potential recovery

---

### 📊 Raw Data Spreadsheet View
**Priority: Medium**

Display raw spectral data in spreadsheet format at the bottom of the Spectrum View tab:

**Requirements**:
- Show wavelengths as columns (or rows)
- Show each file as a row (or column) with intensities
- Scrollable interface (won't fit all wavelengths on screen)
- Synchronized with the plot above - selecting a spectrum highlights its row
- Consider using `st.dataframe()` with pagination or `st.data_editor()` for interactivity

**Layout**:
```
[Plot area - top 60% of screen]
[Data table - bottom 40% of screen]
```

**Implementation Notes**:
- Use pandas DataFrame for data structure
- Consider performance with large datasets (many files × many wavelengths)
- Add option to download the visible data as CSV
- Allow sorting by wavelength or file name

---

### 📂 Row Sets and Column Sets
**Priority: Medium-High**

Add ability to define and save subsets of data for focused analysis:

**Row Sets** (File/Sample Groups):
- Example: "Grass Leaves", "Tree Leaves", "Calibration Samples", "Validation Samples"
- User can assign each file to one or more groups
- Groups can be selected in Preprocessing and Analysis tabs
- Saved with session or exported for reuse

**Column Sets** (Wavelength Ranges):
- Example: "NIR Range (780-2500 nm)", "Visible (380-780 nm)", "Water Absorption Bands"
- User defines named wavelength ranges
- Quick selection in all tabs instead of manually entering min/max
- Could be predefined with common ranges + user custom ranges

**Implementation Notes**:
- Store in session state as dictionaries: `{'row_sets': {...}, 'column_sets': {...}}`
- UI: Dropdown or sidebar section for managing sets
- Export/import functionality (JSON or CSV)
- Validation to prevent empty or invalid sets
- Could integrate with existing wavelength preset buttons (UV, Visible, NIR, Full Range)

---

### 🏷️ Sample Metadata Association
**Priority: High**

Link filenames to sample metadata for richer analysis:

**Current Issue**: Filenames are the only identifier. No way to associate with sample properties or group identifiers.

**Proposed Feature**: Upload a CSV/Excel metadata file with columns:
- `filename` (matches uploaded spectral files)
- `sample_id` (e.g., "Grass 1", "Tree Leaf A")
- Property columns (e.g., `nitrogen_percent`, `moisture_percent`, `species`, `location`)

**Use Cases**:
- PLS regression: Predict `nitrogen_percent` from spectra
- Grouping and coloring plots by `species` or `location`
- Filtering data based on metadata values
- Export combined data (spectra + metadata) for external analysis

**Implementation Notes**:
- Add metadata upload widget in sidebar (separate from spectral file upload)
- Validate that filenames match uploaded spectral files
- Join metadata with `spectral_datasets` in session state
- Display metadata in spreadsheet view alongside spectra
- Use metadata for plot coloring/grouping options
- Store in `SpectralData.metadata` or separate session state dict

**Metadata File Format**:
```csv
filename,sample_id,nitrogen_percent,moisture_percent,species,collection_date
sample1.asd,Grass 1,2.5,60.2,Poa pratensis,2024-01-15
sample2.asd,Grass 2,3.1,58.7,Poa pratensis,2024-01-15
sample3.asd,Tree 1,1.8,45.3,Quercus alba,2024-01-16
```

---

### 🐛 SPC File Reading Issue
**Priority: High**

**Current Problem**: SPC file viewer not producing correct results. Same data in ASD, CSV, and SPC formats should be identical but are not.

**Investigation Needed**:
- Compare known-good CSV data with SPC output side-by-side
- Check wavelength axis (might be wavenumber conversion issue)
- Verify intensity scaling/inversion
- Test with multiple SPC file variants
- Review spectrochempy library documentation for proper usage

**Debugging Steps**:
1. Load same sample in all three formats (ASD, CSV, SPC)
2. Plot overlaid comparison to see discrepancies
3. Check intermediate values (raw binary data, header parsing)
4. Test with known reference SPC files from instrument manufacturer
5. Consider alternative SPC libraries (spc-spectra, galvani) if spectrochempy failing

**Files to Check**:
- `src/file_readers.py:_read_spc()` (lines 217-381)
- Both spectrochempy and manual parsing paths

---

## Spectral Processing Tab

### ✅ Multi-Select for Batch Processing
**Priority: High**

**Current Issue**: Selecting files one-by-one is tedious. Need "Select All" and group selection.

**Proposed Features**:
1. **Select All / Deselect All buttons** above the file multiselect
2. **Select by Row Set**: If row sets are implemented, one-click to select entire group
3. **Inverse Selection**: Select all except current selection
4. **Remember last selection**: Session state keeps previous selection when returning to tab

**Implementation Notes**:
- Add button row above `st.multiselect` widget
- Use session state to track selection
- Integrate with Row Sets feature (see above)

---

### 🔀 Configurable Processing Order
**Priority: Medium-High**

**Current Issue**: Processing steps happen in a fixed order (wavelength → baseline → smooth → derivative → normalize). Users may want different orders.

**Example Use Cases**:
- SNV before derivative vs. derivative before SNV (different results!)
- Smoothing before baseline correction vs. after
- Derivative before wavelength selection (for edge effects)

**Proposed Solution**:
Create a drag-and-drop or numbered list interface where users can reorder processing steps:

```
Processing Pipeline:
[1] Wavelength Selection  [↑↓]
[2] SNV Normalization     [↑↓]
[3] 1st Derivative        [↑↓]
[4] Savitzky-Golay Smooth [↑↓]
```

Or use a dropdown sequence:
```
Step 1: [Dropdown: Wavelength / Baseline / Smooth / Derivative / Normalize]
Step 2: [Dropdown: ...]
...
```

**Implementation Notes**:
- Refactor `SpectralProcessor.process_spectrum()` to accept a list of steps
- Each step is a tuple: `(operation_name, parameters_dict)`
- Process in user-defined order
- Update metadata to include processing sequence
- Add validation to warn about unusual orders (e.g., derivative before smoothing)
- Consider presets: "Standard NIR", "Derivative-first", "Baseline-first"

---

## Advanced Analysis Tab

### 📈 Implement PCA (Principal Component Analysis)
**Priority: High**

**Goal**: Begin implementing PCA once preprocessing functionality is stable.

**Features to Include**:
1. **PCA Computation**:
   - Use scikit-learn `PCA` class (already in requirements.txt)
   - Input: Preprocessed spectral data matrix (n_samples × n_wavelengths)
   - Output: PC scores, loadings, explained variance

2. **Visualization**:
   - **Scores plot**: PC1 vs PC2 (with PC3 option)
   - **Loadings plot**: Show which wavelengths contribute to each PC
   - **Scree plot**: Explained variance by PC
   - **Biplot**: Combined scores and loadings

3. **Interactive Features**:
   - Color points by metadata (species, sample group, etc.)
   - Click on outliers in scores plot to identify which spectrum
   - Select number of components to compute
   - Export PC scores for external analysis

4. **Outlier Detection**:
   - Highlight samples far from cluster centers
   - Hotelling's T² and Q residuals
   - Link back to Spectrum View to remove outliers

**Implementation Steps**:
1. Add PCA section to tab3 in `app.py`
2. Create new `SpectralAnalysis.perform_pca()` method in `src/spectral_processing.py`
3. Ensure all spectra have same wavelength grid (interpolation if needed)
4. Build visualization components with Plotly
5. Integrate with metadata for colored grouping
6. Add export functionality for scores and loadings

**Data Requirements**:
- All spectra must be preprocessed identically
- Same wavelength range for all samples
- Handle missing data appropriately
- Consider scaling options (often done after preprocessing)

**Future Extensions** (after basic PCA):
- PLS (Partial Least Squares) regression
- PLS-DA (Discriminant Analysis) for classification
- Cross-validation for model evaluation
- Prediction for new samples

---

## Technical Debt & Improvements

### 🔧 Performance Optimization
- Lazy loading for large spectral datasets
- Caching of derivative/preprocessing calculations
- Optimize Plotly rendering for many spectra (consider Plotly WebGL)

### 🧪 Testing Expansion
- Add tests for preprocessing functions
- Add integration tests for file reading edge cases
- Create test fixtures with known-good SPC, ASD files

### 📚 Documentation
- Add docstring examples to all public methods
- Create user guide with screenshots
- Add video tutorials for common workflows

---

## Priority Summary

**Immediate (High Priority)**:
1. ✅ Multi-select for batch processing (easy win)
2. 🔄 Plot type toggle in Spectrum View (high value)
3. 🏷️ Sample metadata association (enables PCA)
4. 🐛 Fix SPC file reading (data quality issue)
5. 🎯 Individual spectrum selection & removal (workflow improvement)

**Next Phase (Medium-High Priority)**:
6. 📂 Row sets and column sets (powerful organization)
7. 🔀 Configurable processing order (flexibility)
8. 📈 Implement PCA (core analysis feature)

**Future Enhancement (Medium Priority)**:
9. 📊 Raw data spreadsheet view (nice-to-have)
10. 🧪 Testing and documentation improvements
