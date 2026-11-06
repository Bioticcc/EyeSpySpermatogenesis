# Software evaluation

Reviewed **2026-11-04**, using primary documentation, public repositories and
published methods. **V** = capability verified in documentation/code;
**P** = published biological evidence; **H** = project-specific hypothesis;
**U** = unresolved. The initial review was source-only. The 2026-11-06
overview-first prototype adds restricted TIFF region reads and an optional
SciPy/scikit-image comparator, tested only on synthetic images; no candidate has
been validated on project images. See [the prototype](OVERVIEW_PROTOTYPE.md) and
[Phase 1](PHASE1_IO_VALIDATION.md). “Offline” below means a plausible local runtime
after installation and artifact acquisition,
not a tested disconnected deployment. Installation difficulty is an engineering
estimate. No tool is declared production-ready for mouse PAS/FFPE/40× analysis.

## Provisional priorities

1. **I/O:** characterize with tifffile and cross-check bounded reads with
   Bio-Formats/QuPath; benchmark pyvips if storage/codecs fit. OpenSlide is
   conditional, not a universal OME-TIFF reader.
2. **Isolation:** compare scikit-image/OpenCV morphology with QuPath or ilastik
   pixel classification plus existing instance separation. Fiji provides an
   interactive, macro-scriptable comparator. CellProfiler is a secondary option.
3. **Staging:** inspect STSP first for a Python adapter and SATINN v2 for a PAS/H
   bright-field comparator; neither has demonstrated transfer to these inputs.
   CSS contributes feature concepts. STAGETOOL is a modality-mismatched reference.

These priorities are **H**, not production selections. The owner approved limited
synthetic implementation on 2026-11-06; real-image
Phase 1 evidence and pilot evaluation must precede scientific method selection.

## OME-TIFF and whole-section handling

The [OME-TIFF specification](https://ome-model.readthedocs.io/en/latest/ome-tiff/specification.html)
stores OME-XML with TIFF pixels; conforming sub-resolution pyramids use SubIFDs.
OME-TIFF can be flat or pyramidal and can have multiple series/planes/files.
Extensions and “40×” do not identify layout, calibration or native detail.

| Candidate / primary evidence | V: levels, regions and metadata | Large-image/color limits and project questions |
|---|---|---|
| [Bio-Formats](https://www.openmicroscopy.org/bio-formats/), [WSI API](https://bio-formats.readthedocs.io/en/latest/developers/wsi.html) | Java readers, OME metadata and bounded `openBytes` regions; explicit series/resolution APIs with unflattened pyramids. | Multiple channel/plane types; select the intended series, not a barcode/thumbnail. ROI decoding and optimal storage-tile sizes support bounded work; actual codec/layout costs need measurement. Java API or CLI subprocess from Python, not a native Python reader. |
| [tifffile](https://github.com/cgohlke/tifffile) | Python TIFF/BigTIFF, OME-XML, series/levels and tiled data; documented chunked/Zarr access supports subset reads without an initial whole array. | Parse and validate physical units/axes explicitly. Codecs may require imagecodecs. `asarray`/ordinary `imread` can materialize entire series; memory mapping is not a universal compressed-TIFF solution. Pin compatible tifffile/Zarr APIs, which have changed. |
| [OME ecosystem](https://www.openmicroscopy.org/) | Standards/OME-XML, reference OME Files and chunked OME-Zarr; OMERO provides image management. | An ecosystem, not a single reader dependency. A server is unnecessary for a local application. Conversion to OME-Zarr is an optional derived cache, not required input; validate axes/scale/pixels and include conversion costs. |
| [OpenSlide generic TIFF](https://openslide.org/formats/generic-tiff/), [Python API](https://openslide.org/api/python/) | Fast region reads and level metadata for supported slide formats; Python regions return RGBA with level-0 locations. Generic TIFF detection requires an initial tiled image. | Generic-TIFF detection does not guarantee recognition of every OME SubIFD pyramid, series, physical spacing or arbitrary channels. Check detected levels and OME-XML separately. RGB-oriented interface; alpha/ICC must be handled explicitly. Stripped TIFF is not covered by generic tiled detection. |
| [pyvips](https://github.com/libvips/pyvips), [TIFF loader](https://www.libvips.org/API/8.17/ctor.Image.tiffload.html), [loading behavior](https://www.libvips.org/API/8.16/How-it-opens-files.html) | Python wrapper for demand-driven libvips; tiled TIFF random access, page/SubIFD selection, numeric bands and metadata. | Extract OME physical/plane semantics separately; TIFF resolution fields are not automatically trustworthy OME calibration. Untiled/compressed formats may entail full decompression or temporary disk. Track orientation/ICC, cache and thread behavior. Binary builds vary in codecs/OpenSlide support. |
| [QuPath formats](https://qupath.readthedocs.io/en/stable/docs/intro/formats.html), [exports](https://qupath.readthedocs.io/en/stable/docs/advanced/exporting_images.html) | Whole-slide-aware application with Bio-Formats/OpenSlide backends; ROI original-pixel and pyramidal OME exports. | Compatibility depends on chosen backend/file variant/platform. Verify calibration and selected downsample. Rendered snapshots/overlays are unsuitable as quantitative originals; even “original pixels” saved as JPEG are lossy. Useful D3 integration, not mandatory Python runtime. |

Readers are species/stain/preparation agnostic: file support is **not biological
validation**. There is no fixed magnification/µm-per-pixel requirement in these
general readers. They can retrieve rectangular regions; they do not identify
individual tubules or stage them. QuPath needs a separately developed/reviewed
analysis workflow to produce tubule instances.

### Can the original OME-TIFF be tiled directly?

Often yes: a reader can request analysis rectangles from an already tiled,
codec-supported OME-TIFF. Storage tiles and overlapping analysis regions are
different concepts. Some decoding beyond an ROI is unavoidable, and strip-based
or unusual files may have prohibitive read amplification. Do not promise bounded
RAM solely from a lazy Python handle.

Inspect first; compare the same small ROIs across readers, including corners,
tile seams, levels, dimensions, dtype/channels, orientation, pixel values and
physical spacing. Measure process-tree peak RSS, temporary disk and repeated
random-read time on real inputs. Never read an entire full-resolution array just
to inspect metadata.

If supported, read directly without conversion. If layout/codecs prevent practical
bounded access, propose a validated lossless tiled pyramid or chunked derivative
using Bio-Formats/QuPath/another verified converter. Retain the original, checksum,
conversion command, software version, axes/color/scale mapping and round-trip ROI
checks; fail on unresolved metadata loss. Conversion does not improve optical
resolution. Lack of native detail and lack of pyramids are separate problems.

## Tubule segmentation / image-processing matrix

General tools below have no built-in mouse stage taxonomy. Generic operations
have no validated project-specific FFPE/PAS magnification or µm/pixel setting;
parameters and calibration must be established on the pilot. Export suitability
means a workflow can be assembled, not that a complete cropper exists.

| Tool / primary evidence | V: function, annotations and pretrained status | Whole-slide / individual export / scientific fit |
|---|---|---|
| [Fiji/ImageJ](https://imagej.net/software/fiji/) | Thresholds, morphology, particle/ROI tools and macros; deterministic routes need tuning, not training. No relevant bundled tubule/staging model verified. | Bio-Formats import does not guarantee memory-safe whole-slide processing. Work on bounded regions and export ROI crops/masks through a reviewed macro. H: PAS workflow. |
| [Morphological Segmentation](https://imagej.net/plugins/morphological-segmentation) / [MorphoLibJ](https://imagej.net/plugins/morpholibj) | Morphological gradients and watershed with macro-recordable commands; seeds/tolerance require review. No pretrained model. | Label images can feed standardized instance export; no native whole-slide reconciliation. GUI-recorded commands may retain GUI dependencies in batch/headless mode. H: PAS boundary separation. |
| [QuPath pixel classification](https://qupath.readthedocs.io/en/stable/docs/tutorials/pixel_classification.html) | Interactive pixel labels, intensity/texture features and classification-to-objects workflow. Needs representative training annotations; no project-ready PAS classifier verified. | Whole-slide context and region export; masks, identities and source transforms need adapter checks. Touching objects can remain bridged. H: strongest integrated B/D3 candidate. |
| [ilastik pixel classification](https://www.ilastik.org/documentation/pixelclassification/pixelclassification.html), [headless](https://www.ilastik.org/documentation/basics/headless.html) | Interactive random-forest pixel classification; GUI project training then saved `.ilp` headless/batch prediction and probability export. No PAS model verified. | Feature volume, axes and block/memory settings matter. Use bounded regions; downstream morphology/instance extraction and crop exporter are required. H: limited pixel annotation may improve stain robustness. |
| [CellProfiler object modules](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.8/modules/objectprocessing.html), [batch](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.3/help/other_batch.html) | Modular object identification, separation, measurement/filtering and exported pipelines. Annotation-free deterministic route possible; classifiers are optional, not a mouse staging model. | Prefer pre-extracted regions; a file opening is not proof of WSI-safe decoding. Crop/mask/measurement export needs configured modules and provenance adapter. H: useful alternative, potentially heavier integration. |
| [scikit-image watershed](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html) | Python morphology, labeling, geometry and marker watershed. No training/pretrained staging model. | NumPy-region operations; reader, crop exporter and global reconciliation belong elsewhere. H: compact classical baseline. |
| [OpenCV watershed](https://docs.opencv.org/4.13.0/d3/db4/tutorial_py_watershed.html) | Python/C++ color conversion, thresholding, contours, connected components, morphology and marker watershed. No training required for those operations. | Array-based; not an OME metadata/WSI contract. Explicit RGB/BGR, dtype and boundary-label conversion needed. H: practical baseline alongside scikit-image. |
| [tube_segmentation](https://github.com/pouyaka/tube_segmentation), [paper](https://arxiv.org/html/2301.09887v1) | PyTorch epithelial-layer CNN with prediction/evaluation CLIs. README says datasets and trained networks are available **upon request**, not verified public downloadable checkpoints. | P: formalin/paraffin mink PAS variant, 20× objective; polar-bear H&E validation, not mouse 40× validation. Watershed can separate touching outputs, with manually chosen seeds described. No WSI/OME reader, ready standardized crop export or staging verified. |

The tube-segmentation paper's rendered calibration unit appears inconsistent
with microscopy scale; do not silently turn its reported “0.4 mm/pixel” into a
verified µm/pixel requirement. Resolve against original methods/authors before
scale matching. Its epithelium mask is also not automatically a complete-tubule
mask including lumen. It is a comparative method, not the first implementation.

## Published-method evidence and limits

| Primary study | P: demonstrated method | What it does **not** establish here |
|---|---|---|
| [Benavides et al., PMC4282609](https://pmc.ncbi.nlm.nih.gov/articles/PMC4282609/) | Mouse paraffin sections, H&E at 20×; adaptive equalization, thresholding, morphology, Euclidean distance transform and watershed labels, then shape measurements. | Evidence for classical separation in testis, not validated PAS settings, complete-crop recall or automatic staging. Do not import its size/roundness cutoffs. |
| [Owembabazi et al., PMC12302660](https://pmc.ncbi.nlm.nih.gov/articles/PMC12302660/) | ilastik random-forest segmentation followed by Fiji quantification; paraffin sections and 10× images using hematoxylin, silver and immunohistochemical stains. | Not a published mouse PAS/40× complete-tubule extraction benchmark. Useful tool-chain precedent only. |
| [Gadd et al., PMC13232597 / publisher](https://doi.org/10.1530/RAF-25-0210) | Human and mouse **immunofluorescence**; QuPath ANN-MLP ROI classification with small annotations, cell segmentation/quantification and scripted automation. Mouse touching tubules sometimes bridge. | The supplied research note overstates transfer: this is not PAS bright-field validation. Pixel classification and batch automation are reusable concepts; instance completeness remains unresolved. |

## Staging compatibility matrix

| System / primary evidence | Species, stain, fixation and resolution | Stages, models, inputs and missing evidence |
|---|---|---|
| [STSP](https://github.com/funalab/STSP), [Tokuoka methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12215975/) | P: mouse hematoxylin-PAS, Bouin-fixed/paraffin; acquisition 0.170635 µm/pixel, centered RGB 2040×1536 images. Fixation differs from our formalin contract; counterstain match U. | Twelve stages. Published 1,229-image study; best exact accuracy 79.58%, adjacent tolerance 98.33%, not project estimates. Repository links learned models and data on Google Drive; download integrity/access and artifact terms U. Python test CLI expects image lists, not whole-slide OME. |
| [CSS paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8046964/), [jydada/CSS](https://github.com/jydada/CSS) | P: mouse H&E and concentric histomorphology. Exact formalin/PAS compatibility and physical spacing U in this review. Published total magnifications must not be equated to our objective setting. | Paper groups early/middle/late and subdivides middle stages. The linked repository describes a **later** pipeline: CSNet tubule segmentation at 10×, multi-scale/MTL networks and 204 image-derived features separating I–III from IV–V. Not a deterministic-only system or verified twelve-stage pretrained API. |
| [SATINN / v2 repository](https://github.com/conradlab/SATINN), [v2 methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC13016680/) | V1 is IF; P: v2 uses mouse PAS/hematoxylin bright-field, with paraffin protocols including 10% neutral-buffered formalin among external samples. Example 40× objective, 0.139 µm/pixel; not a calibration prescription. Counterstain/domain matching U. | Twelve stages; paper reports 63% exact and up to 96% within one stage on external validation. Current app offers Brightfield (PAS), raw TIFF and cell-mask inputs, external pretrained networks/demo data. Version-specific bright-field preprocessing and model identity must be checked. |
| [STAGETOOL site](https://stagetool.utu.fi/), [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9780747/) | Mouse DAPI fluorescence, 1024×1024 crops at reported 400× total magnification; not PAS bright-field. Physical spacing and formalin compatibility U here. | Five classes: I–V, VI–VIII, IX, X–XI, XII. Paper describes overlapping whole-section images and merged predictions. Local weights, environment, export API and redistribution terms not verified from site. Comparative design reference, not a PAS inference candidate. |

For CSS, distinguish the [2021 publication record](https://pubmed.ncbi.nlm.nih.gov/33676102/)
from the repository's 2022 implementation; no one-to-one reproduction has been
demonstrated. Retain the morphology concepts while acknowledging its learned
segmentation components. Do not infer all twelve classes from the name “staging”.

### Adapter feasibility checks

STSP's inspected
[WideResNet test config](https://github.com/funalab/STSP/blob/b10e231bcd6785cbf27eb8adc928e39881aa009b/confs/test_wide_resnet.cfg)
uses 896×896 crops, RGB, an external `.npz` checkpoint, `pretrained=False` and GPU
by default; README describes CPU configuration. The
[dataset code](https://github.com/funalab/STSP/blob/b10e231bcd6785cbf27eb8adc928e39881aa009b/src/lib/datasets/dataset.py)
center-crops at test time, applies per-crop min–max normalization and derives
labels from filenames. It asserts minimum crop dimensions and has no zero-range
normalization guard. A future adapter must validate inputs and use a separate
explicit label manifest; it must not infer biological truth from source filenames.
Do not stretch small crops or assume a rectangular export matches expected context.

SATINN's current README requires MATLAB R2026a for the recommended installed-MATLAB
route or MATLAB components for its Windows installer; it calls the app still in
development. It recommends Cellpose-generated cell masks. Source helper
[`setup_tubclass2.m`](https://github.com/conradlab/SATINN/blob/8e0ff33c4d6b553ce60f2be7f2208b0903743010/functions/setup_tubclass2.m)
extracts 2000-pixel objects then downsamples by 0.25. This is evidence about that
helper, not proof that every v2 app/model uses the same preprocessing.
Do not apply v1 IF segmentation helpers to bright-field by assumption. A GUI
demonstration or MATLAB source availability does not establish a stable headless
Python API or bounded whole-slide reader.

## Operations, reproducibility, installation and licensing

“No external app” below still permits compiled libraries. All candidate software
versions/configurations must be pinned in an approved experiment; no compatible
environment has yet been frozen. Code licenses do not automatically establish
dataset/checkpoint redistribution rights. External process use is an engineering
choice, not an exemption from redistribution terms.

| Tool | Python / CLI / batch; external application; offline feasibility | Installation / reproducibility burden | Verified license evidence / unresolved restrictions |
|---|---|---|---|
| Bio-Formats | Java API/CLI; Python subprocess/bridge; JVM required; local operation | Moderate; pin Java, reader/codec build and selected series/resolution | [BSD-2 readers for open formats vs GPL components](https://bio-formats.readthedocs.io/en/latest/developers/components.html); inspect actual bundle, not one blanket license |
| tifffile | Python API/CLI, batch orchestration; no GUI; local | Low–moderate with codecs/Zarr; record versions and XML parsing | [BSD-3-Clause](https://github.com/cgohlke/tifffile/blob/master/LICENSE) |
| OME ecosystem | Depends on component; local formats/OME Files need no server | Select specific component; avoid unnecessary OMERO deployment | [OME Files BSD-2; schemas CC-BY-4.0; other components differ](https://www.openmicroscopy.org/licensing/) |
| OpenSlide | Native/Python APIs, batch; no GUI; local | Moderate, native codecs/library plus binding; pin both | [LGPL-2.1 binding](https://openslide.org/api/python/); audit native bundle |
| pyvips/libvips | Python API and vips CLI; no GUI; local | Moderate, native binary feature set/cache/threads | [pyvips MIT](https://github.com/libvips/pyvips/blob/master/LICENSE.txt), [libvips LGPL-2.1+](https://www.libvips.org/); codec terms separate |
| QuPath | [Headless script CLI](https://qupath.readthedocs.io/en/stable/docs/advanced/command_line.html), Groovy/Java; external runtime; local after setup | Moderate; project/classifier/scripts, extensions, reader and platform versions | [GPL-3.0](https://github.com/qupath/qupath/blob/main/LICENSE); included components separate |
| Fiji/ImageJ | Macros/scripts/batch; external JVM app; local | Moderate; freeze update sites, plugin/JVM versions; headless plugin checks | [ImageJ/Fiji have component-specific terms](https://imagej.net/licensing/), not a single permissive bundle |
| Morphological Segmentation | ImageJ macros; external Fiji/JVM; local | Low added setup in Fiji, but freeze MorphoLibJ/update site and seed/gradient choices | [GPL per plugin documentation](https://imagej.net/plugins/morphological-segmentation); check shipped version |
| ilastik | GUI training then headless CLI/batch; external env/app; local | Moderate–high, project/channel/feature/class ordering and memory/thread settings | [GPL-2.0-or-later with stated extension exception](https://github.com/ilastik/ilastik/blob/main/LICENSE); dependencies differ |
| CellProfiler | Python/CLI headless batch and GUI-authored pipelines; separate environment; local | Moderate–high, scientific/native/reader stack and frozen pipeline | [BSD-3-Clause](https://github.com/CellProfiler/CellProfiler/blob/main/LICENSE); dependency terms separate |
| scikit-image | Python/batch, no external application; local | Low–moderate scientific stack; pin algorithm/array versions and config | [BSD-3-Clause and listed component exceptions](https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt) |
| OpenCV | Python/C++/batch; no external app; local | Low–moderate prebuilt headless wheel; pin build/threading and channel conventions | [Apache-2.0 for current code](https://github.com/opencv/opencv/blob/4.x/LICENSE); bundled codecs/dependencies separate |
| tube_segmentation | Python train/predict/eval CLIs; PyTorch env; local only after models obtained | High relative to baseline; author-request weights/data and reproducibility checks | No root license found in reviewed tree; permission and model/data terms U |
| STSP | Python train/test CLI, CPU configurable; no GUI; local after explicit artifacts | Moderate–high legacy PyTorch stack; checkpoint/config/preprocessing/class order | [MIT code](https://github.com/funalab/STSP/blob/b10e231bcd6785cbf27eb8adc928e39881aa009b/LICENSE); external data/model terms and reachability U; article license is separate |
| CSS | MATLAB/Python components, batch experiments; no verified unified adapter | High; multiple learned stages, missing frozen environment/checkpoint inventory | No root license found; pretrained artifact availability and reuse permission U |
| SATINN/v2 | MATLAB-based app/Windows executable; source functions; Python/headless interface U | High; MATLAB/runtime versions, Cellpose option and external networks; OS matching U | No root license found; code/model/demo rights and MATLAB redistribution terms require checking |
| STAGETOOL | Published analysis workflow; current local/headless/offline reproducibility U in this review | High for our purpose; wrong modality, checkpoint/env access not verified | Site/paper availability does not establish software/model redistribution permission; U |

No listed reader requires expert training labels. For segmentation, A-style
classical tools require parameter development and expert evaluation; pixel
classifiers require small but representative training annotations. Existing
staging checkpoints may avoid initial training, but expert labels remain essential
for transfer validation. CSS/tube segmentation are not ready-to-use pretrained
alternatives merely because their code is public.

## Maintenance snapshot and source anchors

Repository default-head commits inspected below anchor the research-code review.
Dates are commit dates, not support guarantees. General-library dates are GitHub
`pushed_at` snapshots, not releases or proof of quality.

| Repository | Revision / observed activity |
|---|---|
| STSP | `b10e231bcd6785cbf27eb8adc928e39881aa009b`, 2025-07-02; older dependency stack |
| CSS | `60fca8542870f4bf60454947a66acb29d6c991ff`, 2022-10-26; reproducibility/maintenance risk |
| SATINN | `8e0ff33c4d6b553ce60f2be7f2208b0903743010`, 2026-06-07, update 2.0.2 |
| tube_segmentation | `75bec29256e9c4ae520699898ee216cb584f22b7`, 2024-12-09 |
| tifffile / pyvips | Last observed pushes 2026-10-16 / 2026-10-21 |
| QuPath / ilastik | Last observed pushes 2026-11-04 / 2026-10-30 |
| CellProfiler / scikit-image / OpenCV | Last observed pushes 2026-11-04 / 2026-11-03 / 2026-11-04 |
| Bio-Formats / OME / OpenSlide | Current primary documentation inspected (Bio-Formats 8.5; OpenSlide Python 1.4.7); release cadence not audited |
| Fiji / MorphoLibJ | Current docs inspected; exact plugin release/maintenance cadence not audited |
| STAGETOOL | Primary site and paper inspected; current software release/support status U |

Some PMC pages returned bot challenges. Where necessary, indexed primary-paper
text, PubMed or the publisher/arXiv text provided evidence; no local replication
was possible. CSS fixation/calibration and several artifact licenses remain
explicitly unresolved. No third-party weights/datasets were downloaded and no
author contact was initiated. The later overview prototype's exact dependency
versions are recorded in its guide. Real-file compatibility and scientific
comparison remain open; staging still requires separate approval.
