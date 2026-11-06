# Initial separation failure investigation — 2026-11-06

**Finding:** the v0.1 mask represents nearly continuous stained tissue. Its
unconditional hole filling removes enclosed background, including visible spaces
between tubules. Distance-based watershed then operates on the geometry of that
merged mask, without consulting the original RGB image. The current combination
does not establish individual tubules.

This was a diagnostic investigation, not a replacement algorithm or method
selection. Production segmentation and source TIFFs were preserved. Full local
measurements, diagnostic figures and the reproducible investigation script are in
the ignored `outputs/tubule-investigation-2026-11-06-01/` directory; `REPORT.md`
and `index.html` are the entry points.

## Evidence collected

- Reproduced the saved label TIFF **pixel for pixel on all 13 images**, using the
  recorded operations. Source and production-file hashes matched the first trial.
- Measured threshold, closing and hole-fill masks on all 13 images. Immediately
  after thresholding, the largest connected component contains **98.79–99.84% of
  foreground pixels**; after filling, **99.66–99.99%**. These percentages use
  selected foreground as denominator, not the entire image or a reference count.
  Connectivity alone is not an error: watershed can separate connected objects
  when its markers and topography retain the relevant internal structure.
- Performed six controlled cleanup variants, six fixed threshold probes and
  four additional marker-spacing variants on three diagnostic images. Variant
  counts are exploratory, not validated tubule counts or tuned final settings.
- Examined the same 2048×2048 native rectangle at factors 1, 2, 4, 8 and 16 in
  three images. Held the whole-overview threshold and smoothing extent (8 native
  pixels) fixed and omitted closing/filling for that comparison. Also measured
  unsmoothed thresholds. The factor-8 RGB rectangles exactly matched the recorded
  overview crops. This isolates scale from changing field of view/threshold.
- Built a clean synthetic example of four annuli inside an outer ring. Global
  hole filling fills the background separating the annuli as well as their centers.
  This demonstrates the topology assumption absent from the original fixtures.
- Revisited primary method documentation and the previously researched studies.
  Published use of morphology does not validate arbitrary PAS parameters or our
  particular foreground definition.

## What fails and why

1. **The current foreground definition is intensity-only.**
   [Mask generation](../src/eyespy_spermatogenesis/segmentation/overview.py) averages
   RGB channels, smooths and applies a whole-image Otsu threshold. A single darkness
   decision selects walls, cellular epithelium, interstitial staining and artifacts
   together. The result is useful as an approximate tissue mask, not a definition
   of separate tubule interiors. Chromatic differences visible in the original are
   discarded. A displayed `(R+B)/2-G` feature makes some boundaries more apparent
   in the diagnostic regions, but is not calibrated PAS stain deconvolution and
   is not a validated substitute.
2. **Closing has the wrong risk for narrow separators.** Its dilation/erosion can
   bridge narrow background gaps. However, the original threshold already produces
   the giant component. Removing closing alone leaves the main failure, so it is
   a contributor rather than the sole cause.
3. **Unrestricted hole filling is the most destructive cleanup operation.**
   `binary_fill_holes` fills every background region disconnected from the image
   edge. It cannot distinguish a lumen from an enclosed intertubular space or a
   space enclosed by the outer section boundary. In the active review image, it
   adds 211,387 overview pixels; the illustrated central region becomes solid
   foreground despite visible separate tubules in RGB. This destroys internal
   holes/geometry that the next distance transform would otherwise use.
4. **The watershed elevation and markers do not describe tubules.**
   The actual call is `watershed(-distance, markers, mask=foreground)`. Neither
   RGB, a boundary map nor a learned tubule probability reaches this stage.
   Distance maxima after filling describe the bulk section silhouette, with only
   a handful of markers in its giant component. Watershed therefore produces
   section-sized regions and geometric cuts. Reducing marker separation creates
   more arbitrary partitions; it does not supply missing boundary evidence.
5. **Cleanup deletion alone swaps merging for fragmentation.** In the active
   review image, baseline: 70 regions; no closing: 71; no hole fill: 918; neither:
   1,191. Without hole filling, many structures within one tubule become distance
   maxima and receive separate labels. Visual ablations show fragmentation.
   Without smoothing or cleanup, the three probes produce 7,411–12,438 regions.
6. **Higher resolution alone does not fix this representation.** In the active
   review image's fixed native rectangle, a single component contains 98.31% of
   foreground at factor 1 and 98.32% at factor 8 under matched smoothing. Even the
   unsmoothed native threshold retains 95.22% in one component. This is evidence
   about these ROIs and this intensity rule, not proof that factor 8 preserves all
   boundaries. Some other ROI connections do change with scale; small gaps and
   weak walls remain a real resolution risk.

## Why the tests and literature did not establish tubule separation

The existing tests primarily verify source preservation, decoding, coordinates,
exports and predictable behavior on clean annuli/disks. Those fixtures have clear
contrasts, simple backgrounds and no enclosing section capsule. They do not model
dense PAS tissue or establish one-to-one biological instances. Passing the tests
was correctly limited to engineering mechanics, but the first segmentation design
was too generic for the actual object/boundary structure.

The [scikit-image distance-watershed example](https://scikit-image.org/docs/0.25.x/auto_examples/segmentation/plot_watershed.html)
starts with an already correct foreground mask of overlapping disks. Our tissue
mask does not satisfy that example's object-shape assumptions. The same library
also documents [marker watershed using image gradients](https://scikit-image.org/docs/0.25.x/auto_examples/segmentation/plot_marked_watershed.html),
which is a different possible design, not what v0.1 implements.

The [Benavides-Garcia primary methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC4282609/)
used H&E, adaptive histogram equalization, a red/blue intensity combination,
small-object removal, closing, **opening**, hole filling and watershed. Our
mean-RGB/Gaussian prototype is not a reproduction of that procedure, and its H&E
results do not establish our PAS complete-crop performance. Importing that study's
physical/image cutoffs would require independent matching and validation.

The [Gadd QuPath study](https://doi.org/10.1530/RAF-25-0210) used immunofluorescence
and annotated pixel classes with color/intensity, edge, orientation and blob
features. It also reports bridging between touching mouse tubules. Its cell-count
use case does not validate those bridges for our individual-tubule crop requirement.
QuPath/ilastik remain comparison tools, not already validated PAS solutions.

## Consequences and next experiment

Keep the source-coordinate/native-retrieval architecture while testing a new
foreground/boundary and marker design on fixed, expert-reviewed pilot regions:

1. Separate tissue coverage from tubule boundary/interior evidence.
2. Compare color/texture/gradient boundary features with a modest pixel-classifier
   workflow. Define the PAS/counterstain treatment from actual protocol and images;
   do not substitute assumed H&E stain vectors.
3. Generate markers from candidate interiors, wall-enclosed regions and lumen
   evidence together. A lumen-only rule will miss unsuitable/absent-lumen cases.
4. Evaluate marker-controlled separation on boundary evidence. Fill legitimate
   lumens inside established individual instances rather than globally erasing
   all enclosed background before object separation.
5. Verify one-to-one instances, complete native crops and native boundary quality.
   Retain merged, split, ambiguous, artifact and partial cases with reasons.

More regions, visually smoother contours or a count closer to expectation cannot
serve as acceptance criteria. A small expert-reviewed reference subset is needed
to measure merges, splits, misses and complete usable-crop recovery. No accuracy,
recall, specificity, staging eligibility or production winner was established by
this investigation. Biological/scanner metadata, physical calibration, native
optical detail, whole-process resource limits and broader reader agreement remain
separate open checks.
