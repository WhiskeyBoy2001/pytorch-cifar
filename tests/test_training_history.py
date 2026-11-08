import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import main as training
import plot_history


class ConstantClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([2., -1.]))

    def forward(self, inputs):
        return self.logits.expand(inputs.size(0), -1)


class TrainingHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original_cwd = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(os.chdir, self.original_cwd)
        self.dataset = TensorDataset(torch.zeros(4, 1), torch.tensor([0, 0, 0, 1]))

    def run_training(self, *extra_args):
        argv = ['main.py', '--epochs', '2', '--batch-size', '3', '--run-name', 'example']
        with patch.object(sys, 'argv', argv + list(extra_args)), \
                patch.object(training.torch.cuda, 'is_available', return_value=False), \
                patch.object(training.torchvision.datasets, 'CIFAR10', return_value=self.dataset), \
                patch.object(training, 'ResNet18', side_effect=ConstantClassifier), \
                patch.object(training, 'progress_bar'), redirect_stdout(io.StringIO()):
            training.main()

    def read_metrics(self, run_dir):
        with (run_dir / 'metrics.csv').open(newline='', encoding='utf-8') as metrics_file:
            return list(csv.DictReader(metrics_file))

    def test_loss_is_weighted_by_samples_including_partial_batch(self):
        loader = DataLoader(self.dataset, batch_size=3, shuffle=False)
        net = ConstantClassifier()
        criterion = nn.CrossEntropyLoss()
        expected_loss = criterion(net(self.dataset.tensors[0]), self.dataset.tensors[1]).item()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.)
        with patch.object(training, 'progress_bar'), redirect_stdout(io.StringIO()):
            train_loss, train_acc = training.train(0, net, loader, criterion, optimizer, 'cpu')
            test_loss, test_acc = training.test(net, loader, criterion, 'cpu')
        self.assertAlmostEqual(train_loss, expected_loss, places=6)
        self.assertAlmostEqual(test_loss, expected_loss, places=6)
        self.assertEqual(train_acc, 75.)
        self.assertEqual(test_acc, 75.)

    def test_runs_are_separate_and_resume_starts_at_next_epoch(self):
        self.run_training()
        self.run_training('--lr', '0.01')
        run_dirs = sorted(Path('runs').iterdir())
        self.assertEqual(len(run_dirs), 2)
        for run_dir in run_dirs:
            rows = self.read_metrics(run_dir)
            self.assertEqual([int(row['epoch']) for row in rows], [0, 1])
            self.assertEqual(set(rows[0]), {
                'epoch', 'lr', 'train_loss', 'train_acc', 'test_loss', 'test_acc',
            })
            for row in rows:
                self.assertTrue(0 <= float(row['train_acc']) <= 100)
                self.assertTrue(0 <= float(row['test_acc']) <= 100)
            with (run_dir / 'config.json').open(encoding='utf-8') as config_file:
                config = json.load(config_file)
            self.assertEqual(config['batch_size'], 3)
            self.assertEqual(config['epochs'], 2)
            self.assertEqual(config['run_name'], 'example')
            self.assertAlmostEqual(float(rows[0]['lr']), config['lr'])
            self.assertAlmostEqual(float(rows[1]['lr']), config['lr'] / 2)

        checkpoint = torch.load('checkpoint/ckpt.pth', map_location='cpu', weights_only=True)
        self.run_training('--resume', '--epochs', '1')
        resumed_dir = sorted(Path('runs').iterdir())[-1]
        rows = self.read_metrics(resumed_dir)
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]['epoch']), checkpoint['epoch'] + 1)

    def test_completed_epochs_survive_an_interruption(self):
        with patch.object(training, 'test', side_effect=[(0.5, 75.), KeyboardInterrupt]):
            with self.assertRaises(KeyboardInterrupt):
                self.run_training()
        run_dir = next(Path('runs').iterdir())
        rows = self.read_metrics(run_dir)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['epoch'], '0')
        self.assertEqual(float(rows[0]['test_loss']), 0.5)

    def test_plots_can_compare_run_directories_and_csv_files(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest('matplotlib is optional')
        self.run_training()
        self.run_training('--lr', '0.01', '--epochs', '1')
        run_dirs = sorted(Path('runs').iterdir())
        # A run interrupted before its first completed epoch should be skipped.
        empty_dir = Path('runs/empty')
        empty_dir.mkdir()
        (empty_dir / 'metrics.csv').write_text(
            'epoch,lr,train_loss,train_acc,test_loss,test_acc\n', encoding='utf-8')
        with patch.object(sys, 'argv', ['plot_history.py']), redirect_stdout(io.StringIO()):
            plot_history.main()
        self.assertTrue(Path('runs/comparison.png').is_file())
        argv = ['plot_history.py', str(run_dirs[0]), str(run_dirs[1] / 'metrics.csv'),
                '--output', 'selected/comparison.png']
        with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
            plot_history.main()
        self.assertGreater(Path('selected/comparison.png').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
