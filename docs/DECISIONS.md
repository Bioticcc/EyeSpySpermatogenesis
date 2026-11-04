# Architecture decision log

Recorded 2026-11-04. Accepted scope decisions do not select an algorithm.

| ID | Status | Decision and rationale | Alternatives / reopening evidence |
|---|---|---|---|
| 001 | Accepted — owner scope | Mouse, PAS, FFPE, native 40× detail; two input modes. Experimental facts come from owner, not historical filenames. | Broader species/stains/acquisition domains require explicit scope change. |
| 002 | Accepted — owner scope | Separate tubule isolation and staging, including uncertainty. Enables independent tool selection. | Joint detector/stager only as a later evidence-backed comparator. |
| 003 | Accepted — owner scope | Reuse mature software; compare deterministic and lightweight pixel classifiers before new deep learning. | U-Net is an optional historical benchmark, not a default. |
| 004 | Accepted — engineering | Preserve original head on local/remote `legacy`; replace active main tree with a normal descendant commit. Avoid an orphan-main publication that would need history rewriting or branch renaming. | Orphan history considered; unnecessary to achieve clean active architecture. No default-branch/configuration change required. |
| 005 | Accepted — scaffold boundary | Python 3.11+, setuptools, pytest and Ruff; no runtime dependencies or substantive algorithms. | Add trial tools in isolated Phase 1/2 environments after approval; reconsider Python support against their actual constraints. |
| 006 | Proposed | Shared lazy region and tubule-result contracts with source-space transforms and nullable calibration. | Validate actual axes, series, color and export conventions in Phase 1 before coding a schema. |
| 007 | Proposed | Trial tifffile metadata/bounded access with Bio-Formats/QuPath cross-check; pyvips as speed comparator. | Reader choice depends on actual OME-TIFF storage/codecs and measured memory. OpenSlide only after compatibility checks. |
| 008 | Open | Compare D1 direct native tiling, D2 coarse localization with native retrieval, D3 whole-slide integration. | No presumption that 20× localization is more accurate or efficient. |
| 009 | Proposed | A morphology baseline and B QuPath/ilastik/Fiji integrations on common grouped data. | Approve a detailed experiment after representative-image review; no selected thresholds. |
| 010 | Open | Exact I–XII, grouped stages, or both. | Owner/expert must select taxonomy and minimum useful deliverable. |
| 011 | Proposed | Investigate STSP and SATINN v2 before original staging models; CSS-inspired features as independent comparator. | Counterstain, fixation/scale matching, model terms, platform and local validation may change priority. |
| 012 | Accepted — data integrity | Native originals and coordinates persist; no inferred µm/pixel and no upsampling claim of recovered detail. | Unknown/conflicting calibration is explicit and gates physical claims. |
| 013 | Open | Hardware/OS, packaging and project license. | Owner selection; no GUI framework or third-party redistribution decision during Phase 0. |

Future decisions should record date, status, evidence, rationale, alternatives,
owner approval where required and consequences. Supersede entries explicitly;
never silently promote a hypothesis to an accepted implementation choice.
