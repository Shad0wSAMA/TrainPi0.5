import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import pipeline


def write_dataset(root):
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps({
        'codebase_version': 'v3.0', 'total_episodes': 2, 'total_frames': 6,
        'features': {
            'action': {'dtype': 'float32', 'shape': [12]},
            'observation.state': {'dtype': 'float32', 'shape': [12]},
            'observation.images.head': {'dtype': 'video'},
            **{key: {'dtype': 'int64'} for key in ('index', 'episode_index', 'frame_index')},
            'timestamp': {'dtype': 'float32'},
        },
    }))
    (root / 'videos/observation.images.head').mkdir(parents=True)
    (root / 'videos/observation.images.head/file-000.mp4').touch()
    (root / 'meta/tasks.parquet').touch()
    pq.write_table(pa.table({
        'episode_index': [0, 1], 'length': [3, 3],
        'dataset_from_index': [0, 3], 'dataset_to_index': [3, 6],
    }), root / 'meta/episodes/chunk-000/file-000.parquet')
    paths = [root / f'data/chunk-000/file-{i:03d}.parquet' for i in range(2)]
    for i, path in enumerate(paths):
        pq.write_table(pa.table({
            'index': [0, 1, 2], 'episode_index': [i] * 3,
            'frame_index': [0, 1, 2], 'action': [i * 10 + j for j in range(3)],
            'timestamp': [0., 1/30, 2/30],
            'observation.state': [[0.] * 12] * 3,
        }), path)
    return paths


class DatasetIndexTest(unittest.TestCase):
    def test_training_preflight_rejects_bad_indices(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            write_dataset(root)
            with patch.object(sys, 'argv', ['pipeline', 'train', '--dataset', str(root), '--dry-run']):
                with self.assertRaisesRegex(ValueError, 'fix-indices'):
                    pipeline.main()

    def test_dry_run_reports_but_does_not_modify(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            paths = write_dataset(root)
            before = [p.read_bytes() for p in paths]
            with patch.object(sys, 'argv', ['pipeline', 'fix-indices', '--dataset', str(root), '--dry-run']):
                pipeline.main()
            self.assertEqual([p.read_bytes() for p in paths], before)
            self.assertEqual(list(root.rglob('*.before_index_fix')), [])

    def test_repair_handles_file_boundary_inside_episode(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            paths = write_dataset(root)
            combined = pa.concat_tables([pq.read_table(p) for p in paths])
            pq.write_table(combined.slice(0, 4), paths[0])
            pq.write_table(combined.slice(4), paths[1])
            self.assertEqual(pipeline.repair_dataset_indices(root), 2)
            self.assertEqual(pq.read_table(paths[0])['index'].to_pylist(), [0, 1, 2, 3])
            self.assertEqual(pq.read_table(paths[1])['index'].to_pylist(), [4, 5])
            pipeline.inspect_dataset(root)

    def test_invalid_metadata_is_rejected_without_writes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            paths = write_dataset(root)
            meta = root / 'meta/episodes/chunk-000/file-000.parquet'
            table = pq.read_table(meta)
            pq.write_table(table.set_column(2, 'dataset_from_index', pa.array([0, 4])), meta)
            before = [p.read_bytes() for p in paths]
            with self.assertRaisesRegex(ValueError, 'episode metadata'):
                pipeline.repair_dataset_indices(root)
            self.assertEqual([p.read_bytes() for p in paths], before)

    def test_guard_rejects_episode_local_indices(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            write_dataset(root)
            with self.assertRaisesRegex(ValueError, 'fix-indices'):
                pipeline.validate_dataset_indices(root)

    def test_repair_preserves_payload_backup_and_correct_tail_sampling(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            paths = write_dataset(root)
            before = paths[1].read_bytes()
            payload = pq.read_table(paths[1]).drop(['index'])
            self.assertEqual(pipeline.repair_dataset_indices(root), 1)
            self.assertEqual(pq.read_table(paths[1])['index'].to_pylist(), [3, 4, 5])
            self.assertTrue(pq.read_table(paths[1]).drop(['index']).equals(payload))
            self.assertEqual(paths[1].with_name(paths[1].name + '.before_index_fix').read_bytes(), before)
            self.assertFalse(paths[0].with_name(paths[0].name + '.before_index_fix').exists())
            pipeline.validate_dataset_indices(root)
            self.assertEqual(pipeline.repair_dataset_indices(root), 0)
            # Reader's boundary rule, with distinct action values in each episode.
            indices = pq.read_table(paths[1])['index'].to_pylist()
            actions = [0, 1, 2, 10, 11, 12]
            windows = [[actions[max(3, min(5, index + d))] for d in range(3)] for index in indices]
            self.assertEqual(windows, [[10, 11, 12], [11, 12, 12], [12, 12, 12]])

    def test_invalid_episode_alignment_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            paths = write_dataset(root)
            table = pq.read_table(paths[1])
            pq.write_table(table.set_column(2, 'frame_index', pa.array([0, 2, 1])), paths[1])
            before = [p.read_bytes() for p in paths]
            with self.assertRaisesRegex(ValueError, 'frame_index'):
                pipeline.repair_dataset_indices(root)
            self.assertEqual([p.read_bytes() for p in paths], before)
            self.assertEqual(list(root.rglob('*.before_index_fix')), [])

    def test_existing_backup_blocks_repair_without_overwrite(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            paths = write_dataset(root)
            backup = paths[1].with_name(paths[1].name + '.before_index_fix')
            backup.write_bytes(b'original backup')
            with self.assertRaisesRegex(ValueError, 'Backup already exists'):
                pipeline.repair_dataset_indices(root)
            self.assertEqual(backup.read_bytes(), b'original backup')
            self.assertEqual(pq.read_table(paths[1])['index'].to_pylist(), [0, 1, 2])


if __name__ == '__main__':
    unittest.main()

