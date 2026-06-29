# Oval Defect Detection

Traditional computer-vision prototype for detecting Fab AOI `oval defect`
patterns: oval shadow rings or oval shadow blobs.

## Install

```bash
pip install -r requirements.txt
```

## Windows Folder Layout

```text
C:\defect_ai\
    bad\
    good\
    unknown\
    output\
```

## Run

```bash
python src\detect_oval_defect.py --root C:\defect_ai
```

The script reads `bad`, `good`, and `unknown`, writes debug images and metrics
to `output`, and first validates whether `bad` and `good` are separable before
hard-labeling unknown images.

## Current Validation Result

The sample run exported from Photos contained:

- `bad`: 48 images
- `good`: 50 images
- `unknown`: 98 images

Current traditional CV metrics did **not** reliably separate bad/good samples:

- validation status: `FAILED`
- bad recall: `0.0208`
- good specificity: `1.0000`
- balanced accuracy: `0.5104`

Because validation failed, unknown images are marked `unvalidated` instead of
being forced into production labels. See `sample_run/validation` and
`sample_run/metrics` for details.

## Notes

The detector intentionally avoids gold-color-only thresholding and hard pad
masking. It uses Lab-L illumination correction, horizontal stripe suppression,
multi-scale dark shadow response, ellipse/ring fitting, and structure penalties.

More samples are needed, especially:

- weak oval defects
- oval defects in dark regions
- oval defects in gold regions
- oval defects crossing gold/dark boundaries
- good hard negatives with smooth non-oval shadows
- good hard negatives with dense pad/trace/black-line structures
