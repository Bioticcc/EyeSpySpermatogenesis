# Experimental specifications

Owner-confirmed design contract, recorded 2026-11-04. Historical files and
third-party training protocols cannot override this table.

| Parameter | Confirmed specification |
|---|---|
| Species | Mouse |
| Stain | PAS (Periodic Acid–Schiff) |
| Preparation | FFPE: formalin-fixed, paraffin-embedded |
| Primary magnification | 40× |
| Input A | Large OME-TIFF containing a complete testis tissue section |
| Input B | Folder of existing 40× microscopy images |
| Analysis target | Individual tubules retaining native 40×-equivalent detail |
| Optional coarse processing | Approximately 20×, or another suitable lower-resolution level, for localization only; benchmark before selection |

## Information still needed

Every entry below is **PENDING CONFIRMATION**. A blank metadata field must stay
unknown; use `null` with an explicit reason in future machine-readable records.

| Detail | How to establish it / why it matters |
|---|---|
| Hematoxylin or other PAS counterstain | Obtain staining protocol; determines relevance of He-PAS/PAS-H classifiers |
| Scanner/microscope manufacturer and model | Acquisition records; reader support and domain shift |
| OME-TIFF pyramidal status and available levels | Inspect series, IFD/SubIFD structure and associated images |
| Physical X/Y pixel spacing at each usable level | OME-XML, validated vendor metadata or independently calibrated measurement; physical filters and model compatibility |
| Meaning of 40× | Objective, nominal scanner setting, or other convention; do not infer calibration |
| Native dimensions, storage tiles/strips | Inspect headers; bounded decoding costs |
| Bit depth, channel axes/order, photometric interpretation, color/ICC profile | Inspect headers and metadata; avoid silent biological/color changes |
| Folder-image formats: TIFF, PNG, JPEG, mixed, or other | Inventory representative files and compression history |
| Typical tubules per 40× frame | Expert image review; no inherited top-K cap |
| Number of labs/scanners/acquisition sources | Sample manifest; evaluation grouping and generalization |
| Section thickness | Histology protocol; visible cell organization |
| Mouse strain and age | Owner records; biological scope |
| Experimental conditions and expected pathology | Owner records; avoid rejecting meaningful abnormal tubules |
| Expert tubule annotations: availability and format | Annotation inventory and permission review |
| Expert stage labels: availability and format | Independent expert review and taxonomy agreement |
| Initial exact I–XII stages, grouped stages, or both | Owner/expert deliverable decision |
| Hardware, RAM, OS, and deployment/package requirements | Target-system inventory; practical reader/application limits |

Physical pixel size is separate from optical resolving power and nominal
magnification. Check metadata units and provenance and reconcile disagreements;
neither “40×” nor a familiar frame size implies a particular µm/pixel value.
Pyramid dimensions alone do not prove native optical detail. Request acquisition
records when that provenance is absent.

OME-TIFF is a metadata/storage format, not a guarantee of pyramids, RGB data, or
efficient rectangular reads. Identify the actual series/plane and storage layout.
A 20× localization result must map back to a genuine native-detail source region.
If resolution is insufficient, mark the input `insufficient_native_detail`; do
not manufacture eligibility by upsampling. If resolution is unverified, mark it
`resolution_unverified` and allow characterization only until resolved.
