"""Numbered exports preserve earlier independent and negotiated compositions."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

from composition_results import reserve_result
from composer_b import negotiate


class ResultFolderTests(unittest.TestCase):
    def test_independent_counters_and_existing_results_are_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = reserve_result('A', root)
            (first / 'composition.wav').write_bytes(b'previous audio')
            self.assertEqual(reserve_result('B', root).name, 'B_1')
            self.assertEqual(reserve_result('A', root).name, 'A_2')
            (root / 'A_9').mkdir()
            self.assertEqual(reserve_result('A', root).name, 'A_10')
            self.assertEqual((first / 'composition.wav').read_bytes(), b'previous audio')

    def test_parallel_reservations_do_not_overwrite(self):
        with TemporaryDirectory() as directory:
            with ThreadPoolExecutor(max_workers=4) as executor:
                paths = list(executor.map(lambda _: reserve_result('AB', directory), range(12)))
            self.assertEqual(len(set(paths)), 12)
            self.assertEqual({p.name for p in paths}, {f'AB_{i}' for i in range(1, 13)})

    def test_negotiation_cli_allocates_new_folder_for_each_song(self):
        with TemporaryDirectory() as directory:
            with patch.object(negotiate, 'run') as run:
                for _ in range(2):
                    with patch.object(sys, 'argv', ['composer_b.negotiate', '-p', 'piano', '-o', directory]):
                        negotiate.main()
            self.assertEqual([call.args[1].name for call in run.call_args_list], ['AB_1', 'AB_2'])


if __name__ == '__main__':
    unittest.main()
