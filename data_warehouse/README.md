# Football Data Warehouse

Downloads historical football datasets into a **versioned raw data lake**
under `data/raw/`, with recorded provenance (source URL, download time,
checksum, file size) per version. This module only acquires and stores raw
data — it contains no prediction, normalization, or machine learning logic.
(Normalization, validation, and a canonical store are planned as separate,
later subsystems.)

## Requirements

Python 3.9+, `requests`, `PyYAML` (see `requirements.txt`).

## Usage

```
python -m data_warehouse.cli download --source football_data_co_uk
python -m data_warehouse.cli download --source football_data_co_uk --leagues E0 SP1 --seasons 2324 2223
python -m data_warehouse.cli download --source football_data_co_uk --force
```

## Versioned raw lake layout

Every dataset gets its own directory holding one immutable subdirectory per
version, plus a pointer to the latest one:

```
data/raw/football_data_co_uk/E0/2324/
  latest.json                      # {"latest_version": "20260706T140512123456Z"}
  20260706T140512123456Z/
    2324.csv
    2324.csv.meta.json
```

The metadata sidecar records:

```json
{
  "source": "football_data_co_uk",
  "identifier": "E0/2324",
  "source_url": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
  "version": "20260706T140512123456Z",
  "local_path": "...",
  "downloaded_at_utc": "2026-07-06T14:05:12+00:00",
  "file_size_bytes": 12345,
  "checksum_sha256": "..."
}
```

The checksum is computed by this tool at download time (football-data.co.uk
does not publish official checksums) — treat it as a fingerprint for detecting
future drift, not third-party-verified integrity data.

A version directory, once written, is never mutated or deleted by this
module. This matters because a "current season" CSV on football-data.co.uk is
appended to weekly — without versioning, a later re-download would silently
overwrite the data a backtest was run against. With versioning, any past
snapshot can still be read, so training/backtesting code can pin an explicit
version instead of always reading "whatever is there now."

## Fetch behavior (skip / new version / unchanged)

- **No version exists yet** → always fetch and write version 1.
- **A version exists, `force=False`** (the default) → skipped entirely, no
  network request is made. This is the literal "never overwrite unless
  explicitly requested" behavior — re-checking a source for updates is itself
  the explicit request.
- **A version exists, `force=True`** → the source is re-fetched. If the
  content is byte-identical (via SHA-256) to the latest existing version, no
  new version is created (`status: "unchanged"`) — this keeps the lake from
  filling up with redundant snapshots of closed seasons that never change.
  If the content differs, a new version directory is written and the latest
  pointer advances; the old version is left untouched.

## Design

- `ingest/base_source.py` — `BaseDataSource`, the abstract interface every
  source implements. The shared skip / new-version / unchanged logic lives
  here so individual sources only need to describe *what* is downloadable
  and *how* to reach it.
- `ingest/http_client.py` — a rate-limited, retrying HTTP client shared by
  all sources.
- `ingest/metadata_store.py` — manages the versioned directory layout:
  writing new version snapshots, reading/advancing the `latest.json`
  pointer, and reading a version's metadata sidecar.
- `ingest/models.py` — `DatasetRef`/`DatasetMetadata`/`DownloadOutcome`, the
  raw-lake data contract. (Later subsystems — normalize, validate, canonical
  — will define their own schemas elsewhere; this one is scoped to raw
  ingestion only.)
- `ingest/exceptions.py` — the exception hierarchy for this subsystem.
- `sources/` — one module per external source. `football_data_co_uk.py` is
  the first; it only knows the URL pattern
  (`{base_url}/{season}/{league_code}.csv`) and local layout.
- `config/config.yaml` — every URL, league code, season, and HTTP setting.
  Never hardcode these in source code.
- `cli.py` — argument parsing and orchestration only; no business logic.

## Adding a new source

1. Add its configuration section to `config/config.yaml` and a matching
   dataclass + parsing in `config/loader.py`.
2. Create `sources/<new_source>.py` implementing `BaseDataSource`
   (at minimum, `list_available_datasets`).
3. Register it in `registry.py` (`SOURCE_NAMES` and `build_source`).
4. Add unit tests mirroring `tests/test_football_data_co_uk_source.py`,
   mocking the HTTP client so tests never hit the network.

## Running tests

```
python -m unittest discover -s data_warehouse/tests
```
