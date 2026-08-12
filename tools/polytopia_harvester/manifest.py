from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional


MANIFEST_FIELDS = (
    "index",
    "filename",
    "harvested_at",
    "cycle_seconds",
    "size_bytes",
    "source_filename",
    "sha256",
    "source_modified_at",
)


class DatasetError(RuntimeError):
    pass


class DuplicateStateError(DatasetError):
    def __init__(self, sha256: str, existing_filename: str) -> None:
        super().__init__(
            f"exact duplicate save detected: SHA-256 {sha256} already belongs to {existing_filename}"
        )
        self.sha256 = sha256
        self.existing_filename = existing_filename


class CopyIntegrityError(DatasetError):
    pass


@dataclass(frozen=True)
class HarvestRecord:
    index: int
    filename: str
    harvested_at: str
    cycle_seconds: float
    size_bytes: int
    source_filename: str
    sha256: str
    source_modified_at: str

    def as_row(self) -> dict:
        return {
            "index": int(self.index),
            "filename": self.filename,
            "harvested_at": self.harvested_at,
            "cycle_seconds": f"{float(self.cycle_seconds):.3f}",
            "size_bytes": int(self.size_bytes),
            "source_filename": self.source_filename,
            "sha256": self.sha256,
            "source_modified_at": self.source_modified_at,
        }


def state_filename(index: int) -> str:
    if int(index) < 1:
        raise ValueError("map index must be >= 1")
    return f"map_{int(index):06d}.state"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class DatasetManager:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.raw_dir = self.output_dir / "raw_states"
        self.manifest_path = self.output_dir / "harvest_manifest.csv"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._hash_to_filename = self._load_hash_index()

    def state_files(self) -> list[Path]:
        return sorted(
            path
            for path in self.raw_dir.glob("map_*.state")
            if re.fullmatch(r"map_\d{6,}\.state", path.name) and path.is_file()
        )

    def existing_count(self) -> int:
        return len(self.state_files())

    def existing_indices(self) -> list[int]:
        indices = []
        for path in self.state_files():
            stem = path.stem
            try:
                indices.append(int(stem.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return sorted(indices)

    def next_index(self) -> int:
        indices = self.existing_indices()
        return (indices[-1] + 1) if indices else 1

    def remaining_to_target(self, target_count: int) -> int:
        return max(0, int(target_count) - self.existing_count())

    def _manifest_rows(self) -> list[dict]:
        if not self.manifest_path.exists():
            return []
        with self.manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))

    def historical_cycle_times(self) -> list[float]:
        values = []
        for row in self._manifest_rows():
            try:
                value = float(row.get("cycle_seconds", ""))
                if value > 0:
                    values.append(value)
            except (TypeError, ValueError):
                continue
        return values

    def _load_hash_index(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        represented = set()
        for row in self._manifest_rows():
            filename = str(row.get("filename", "") or "")
            digest = str(row.get("sha256", "") or "").lower()
            if filename and len(digest) == 64:
                represented.add(filename)
            if filename and len(digest) == 64:
                result[digest] = filename

        for path in self.state_files():
            if path.name in represented:
                continue
            try:
                result[sha256_file(path)] = path.name
            except OSError:
                continue
        return result

    def is_duplicate_hash(self, digest: str) -> Optional[str]:
        return self._hash_to_filename.get(digest.lower())

    def append_manifest(self, record: HarvestRecord) -> None:
        new_file = not self.manifest_path.exists() or self.manifest_path.stat().st_size == 0
        with self.manifest_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(record.as_row())
            handle.flush()
            os.fsync(handle.fileno())

    def store_unique_state(
        self,
        source: Path,
        index: int,
        cycle_seconds: float,
        harvested_at: Optional[datetime] = None,
    ) -> HarvestRecord:
        source = source.resolve()
        source_digest = sha256_file(source)
        duplicate = self.is_duplicate_hash(source_digest)
        if duplicate is not None:
            raise DuplicateStateError(source_digest, duplicate)

        destination = self.raw_dir / state_filename(index)
        if destination.exists():
            raise DatasetError(f"refusing to overwrite existing harvested file: {destination}")
        temporary = destination.with_suffix(destination.suffix + ".partial")
        if temporary.exists():
            raise DatasetError(f"stale partial destination requires inspection: {temporary}")

        try:
            shutil.copy2(source, temporary)
            if not temporary.exists() or temporary.stat().st_size <= 0:
                raise CopyIntegrityError(f"copied save is empty or missing: {temporary}")
            copied_digest = sha256_file(temporary)
            source_digest_after = sha256_file(source)
            if copied_digest != source_digest or source_digest_after != source_digest:
                raise CopyIntegrityError(
                    "source changed during copy or copied SHA-256 did not match the stable source"
                )
            temporary.replace(destination)
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise

        timestamp = harvested_at or datetime.now().astimezone()
        source_mtime = datetime.fromtimestamp(source.stat().st_mtime).astimezone()
        record = HarvestRecord(
            index=int(index),
            filename=destination.name,
            harvested_at=timestamp.isoformat(timespec="seconds"),
            cycle_seconds=float(cycle_seconds),
            size_bytes=int(destination.stat().st_size),
            source_filename=source.name,
            sha256=source_digest,
            source_modified_at=source_mtime.isoformat(timespec="seconds"),
        )
        try:
            self.append_manifest(record)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        self._hash_to_filename[source_digest] = destination.name
        return record
