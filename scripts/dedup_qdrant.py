#!/usr/bin/env python3
"""
Deduplicate Qdrant collection by metadata.arxiv_id.

For each arxiv_id with multiple points, keeps the one whose UUID matches
the deterministic id (uuid5) used by build_vector_db.py, and deletes the rest.
If no point matches the deterministic id, keeps the first one encountered.

Usage:
    python scripts/dedup_qdrant.py                        # dry-run (default)
    python scripts/dedup_qdrant.py --apply                # actually delete duplicates
    python scripts/dedup_qdrant.py --qdrant_url http://localhost:6433
"""

import argparse
import uuid
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

# Must match build_vector_db.py
_QDRANT_NS = uuid.UUID("a3f1b2c4-d5e6-7890-abcd-ef1234567890")
COLLECTION_NAME = "paper_knowledge_base"
SCROLL_BATCH = 256


def _deterministic_id(key: str) -> str:
    return str(uuid.uuid5(_QDRANT_NS, key))


def main():
    parser = argparse.ArgumentParser(description="Deduplicate Qdrant collection")
    parser.add_argument("--qdrant_url", default="http://localhost:6433")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--apply", action="store_true", help="Actually delete duplicates (default is dry-run)")
    args = parser.parse_args()

    client = QdrantClient(url=args.qdrant_url, timeout=300)
    info = client.get_collection(args.collection)
    print(f"Collection: {args.collection}")
    print(f"Total points before: {info.points_count}")

    # Phase 1: scan all points, group by arxiv_id
    arxiv_to_points = defaultdict(list)  # arxiv_id -> [(point_id, ...)]
    offset = None
    scanned = 0

    print("Scanning points...")
    while True:
        results, next_offset = client.scroll(
            collection_name=args.collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            break
        for point in results:
            meta = point.payload.get("metadata", {})
            arxiv_id = meta.get("arxiv_id") or meta.get("paper_id", "")
            if arxiv_id:
                arxiv_to_points[arxiv_id].append(str(point.id))
        scanned += len(results)
        if scanned % 10000 < SCROLL_BATCH:
            print(f"  Scanned {scanned} points, {len(arxiv_to_points)} unique arxiv_ids...")
        offset = next_offset
        if offset is None:
            break

    print(f"Scan complete: {scanned} points, {len(arxiv_to_points)} unique arxiv_ids")

    # Phase 2: find duplicates
    to_delete = []
    dup_groups = 0
    for arxiv_id, point_ids in arxiv_to_points.items():
        if len(point_ids) <= 1:
            continue
        dup_groups += 1
        # Prefer the deterministic UUID
        det_id = _deterministic_id(arxiv_id)
        if det_id in point_ids:
            keep = det_id
        else:
            keep = point_ids[0]
        for pid in point_ids:
            if pid != keep:
                to_delete.append(pid)

    print(f"Duplicate groups: {dup_groups}")
    print(f"Points to delete: {len(to_delete)}")
    print(f"Points to keep:   {scanned - len(to_delete)}")

    if not to_delete:
        print("No duplicates found.")
        client.close()
        return

    if not args.apply:
        print("\nDry-run mode. Re-run with --apply to delete duplicates.")
        client.close()
        return

    # Phase 3: delete in batches
    DELETE_BATCH = 500
    print(f"\nDeleting {len(to_delete)} duplicate points...")
    for i in range(0, len(to_delete), DELETE_BATCH):
        batch = to_delete[i:i + DELETE_BATCH]
        client.delete(
            collection_name=args.collection,
            points_selector=PointIdsList(points=batch),
        )
        done = min(i + DELETE_BATCH, len(to_delete))
        if done % 5000 < DELETE_BATCH:
            print(f"  Deleted {done}/{len(to_delete)}")

    # Verify
    info_after = client.get_collection(args.collection)
    print(f"\nDone! Points after dedup: {info_after.points_count}")
    client.close()


if __name__ == "__main__":
    main()
