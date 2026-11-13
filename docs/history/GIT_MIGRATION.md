# Git migration and preservation record

Date: **2026-11-04**. Original repository:
<https://github.com/Bioticcc/EyeSpySpermatogenesis>.

## Initial inspection

The supplied workspace `EyeSpySpermatogenesisRebooted` was not a Git checkout:
`.git` was empty/protected. No usable local branch, tags, remotes, history or index
existed. Its only ordinary file was the owner's research note,
`EyeSpySpermatogenesisReboot.md` (23,034 bytes), SHA-256:

```text
7a436e946580d9b6fc633c8ba4536e50b7387a42a65f66ba32517155323e86b8
```

`.agents`, `.codex` and `.aws` were empty protected directories. There were no
local image datasets, weights, generated outputs or credential-looking files in
this initial workspace inventory. No ignore rules applied before initialization.
This is a scoped filename/file inventory, not a comprehensive secret audit of
all Git history or neighboring directories.

The original repository was inspected remotely and through a temporary bare
clone. At inspection, GitHub's default branch was `main`, only `main` existed,
and no tags were advertised. Its eight commits ended at:

```text
4af472c53beb3810828923ddd252f451374d2a01
```

The original root tree is:

```text
d49ce525f887e68f5dda75f7ec07dcf742b7d877
```

The tree has **496 tracked files**: 391 under `data`, 7 under `models`, 30 under
`outputs`, 43 under `src`, 11 under `tests`, and 14 elsewhere. Data includes 187
annotation JSON files plus their placeholder, one labels CSV and 202 raw files
(189 TIFFs and 13 PNGs). Models include six `.pt` files and a placeholder.
Tracked generated outputs and images remain historical assets; none were copied
into the new active package. No credential-like filenames were found in the
inspected tracked path list; contents/history were not exhaustively secret-scanned.

Remote Git cannot reveal ignored/untracked files from an old machine. Those were
**not available for inventory or preservation here**. Before retiring any old
checkout, its owner must separately inspect `git status --short --untracked-files=all`
and `git status --short --ignored` and privately back up wanted local-only files.
Do not commit private files merely to make an archive complete.

## Migration method and consequences

1. Create remote `legacy` at the exact original head; verify its SHA matches old
   `main`. The GitHub connector returned HTTP 403 for branch creation, so ordinary
   authenticated Git provided the supported fallback. No GitHub settings changed.
2. Initialize the workspace with authorized Git metadata access; fetch the original
   history and create local `legacy` tracking `origin/legacy`. The initial partial
   fetch avoided materializing historical assets into the working tree.
3. Fetch the existing historical objects fully for archive integrity checks. These
   are original tracked repository assets, not new third-party research datasets
   or pretrained staging models. Check missing-object counts and `git fsck`.
4. Build the replacement scaffold with an empty index, using the original commit
   as the new `main` commit's parent. Keep all old objects/history and the immutable
   archival tip on `legacy`; do not create a `legacy/` directory on main.
5. Commit only reviewed new project paths, then publish by a **normal fast-forward
   push** after verification. The initial push required additional OAuth `workflow`
   permission. After owner-approved authorization, publication succeeded from
   `4af472c` to `45ddcfa`; remote legacy remained at the exact original head.

An orphan branch was considered and rejected: it would require an unnecessary
remote-history replacement or branch/default-name migration. A clean active file
tree does not require erasing ancestry. Old tracked files disappear from the
current `main` snapshot but remain recoverable on `legacy` and in Git history.
This reset does not adopt their architecture. No remote branch deletion,
force-push, tag rewrite or default-branch change is needed or authorized.

## Local-only and ignored inventory

The owner's research note remains unchanged on disk. It is excluded **locally**
through `.git/info/exclude`, not committed as current design authority. The
attachment containing the assignment remains outside the repository. Generated
`.venv`, egg-info and test/lint caches are ignored development artifacts.
Private acquisition files, annotations, model downloads and results must remain
outside tracked source unless separately reviewed and authorized.

## Recovery and verification

These commands are read-only and should identify the original snapshot:

```bash
git rev-parse legacy
git rev-parse 'legacy^{tree}'
git rev-list --count legacy
git ls-tree -r --name-only legacy
git fsck --full
git ls-remote --heads --tags origin
```

Expected archival results: the commit/tree above, eight commits, 496 paths and
remote `refs/heads/legacy` at the original SHA. To inspect or run historical code,
use a separate clone/worktree and an isolated legacy environment. For example,
outside this checkout:

```bash
git clone --branch legacy --single-branch \
  https://github.com/Bioticcc/EyeSpySpermatogenesis.git EyeSpySpermatogenesis-legacy
```

Cloning is not permission to publish private files or execute old model loaders.
Verify artifact provenance before any optional benchmark. No external staging
weights or datasets were downloaded for Phase 0.

## Verification results

The final migration result and exact validation evidence are recorded in
[VALIDATION.md](../VALIDATION.md). No additional archival tag is needed: `legacy`
names the original complete tracked snapshot. Future agents must preserve that
ref. Any future history rewrite or default-branch change requires explicit owner
approval and a separate reviewed procedure.

## Publication completed

On 2026-11-04, the owner approved GitHub authorization with the required `workflow`
scope. The normal command `git push --set-upstream origin main` succeeded,
publishing `45ddcfa1a06166e4f476e728525e27c0ac5de7e9` and setting local main to
track origin/main. The clean 28-file scaffold and CI workflow are now published.
Later documentation commits record this completed migration.

Remote `legacy` remains `4af472c53beb3810828923ddd252f451374d2a01`. No force-push,
remote deletion, tag rewrite or default-branch change was performed. No GitHub
authorization or configuration steps remain for this migration. Hosted CI
evidence is recorded in [VALIDATION.md](../VALIDATION.md).

## History cleanup (2026-11-13)

The owner approved removing lab data from all history of `main` and `legacy`
(the repository is public). `git filter-repo` (2.47.0) dropped every file under
`data/raw/` and `data/labels/` and every `.pt` weight file, except:

- `data/raw/Image_11.tif` and `data/raw/Multistage oct 14_63.tif` (examples)
- their annotations in `data/labels/annotations/`
- the legacy overlays in `outputs/seg_viz/` and `outputs/mask_debug/`, the
  logo PNG and the README overlay figures

The `main` file tree was unchanged. Reachable history went from about 1 GB to
about 47 MB. Every commit SHA changed, so SHAs earlier in this record and in
[VALIDATION.md](../VALIDATION.md) refer to the pre-cleanup history:

| Commit | Before | After |
|---|---|---|
| Legacy head | `4af472c` | `33c1dc4` |
| Scaffold reset | `38984ab` | `03ba2d0` |
| First published main | `45ddcfa` | `166a8c2` |
| Last commit before cleanup | `89b3a6d` | `f39f23b` |

Both branches were force-pushed with owner approval. The full raw data stays
local in `Inputs/` and is never committed.
