'''Compare per-epoch metrics saved by main.py.'''
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


METRICS = ('train_loss', 'test_loss', 'train_acc', 'test_acc')


def load_history(path):
    path = Path(path)
    csv_path = path / 'metrics.csv' if path.is_dir() else path
    with csv_path.open(newline='', encoding='utf-8') as metrics_file:
        reader = csv.DictReader(metrics_file)
        required = {'epoch', *METRICS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('%s is missing metric columns' % csv_path)
        rows = [{key: float(row[key]) for key in required} for row in reader]
    config_path = csv_path.with_name('config.json')
    label = csv_path.parent.name
    if config_path.exists():
        with config_path.open(encoding='utf-8') as config_file:
            label = json.load(config_file).get('run_name') or label
    return csv_path, label, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', help='run directories or metrics.csv files')
    parser.add_argument('--log-dir', default='runs', help='scan this directory if no paths are given')
    parser.add_argument('--output', default='runs/comparison.png', help='output image path')
    parser.add_argument('--show', action='store_true', help='also open an interactive plot window')
    args = parser.parse_args()
    paths = args.paths or sorted(Path(args.log_dir).glob('*/metrics.csv'))
    if not paths:
        parser.error('no metrics.csv files found; train a model first')

    histories = []
    for path in paths:
        try:
            history = load_history(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        if history[2]:
            histories.append(history)
        else:
            print('Skipping %s: no completed epochs.' % history[0])
    if not histories:
        parser.error('no completed epochs to plot')

    try:
        import matplotlib
        if not args.show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        parser.error('plotting requires matplotlib: python -m pip install matplotlib')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    label_counts = Counter(label for _, label, _ in histories)
    for csv_path, label, rows in histories:
        if label_counts[label] > 1:
            label = '%s (%s)' % (label, csv_path.parent.name)
        epochs = [row['epoch'] for row in rows]
        for ax, metric in zip(axes.flat, METRICS):
            ax.plot(epochs, [row[metric] for row in rows], label=label,
                    marker='o' if len(rows) == 1 else None)

    for ax, metric in zip(axes.flat, METRICS):
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xlabel('Epoch (zero-based)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel('Accuracy (%)' if metric.endswith('acc') else 'Loss')
        if metric.endswith('acc'):
            # Leave room for curves at exactly 0% or 100% to remain visible.
            ax.set_ylim(-2, 102)
        ax.grid(alpha=0.25)
        ax.legend()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=150)
    print('Saved comparison plot to: %s' % output)
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
