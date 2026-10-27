"""Render the public figures. Metric plots read assets/figures/results.json; no dataset images or inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/figures'
data = json.loads((OUT / 'results.json').read_text())
NAMES = {'S0': 'CNN-BiGRU, fixed pooling (S0)',
         'S0-A': 'CNN-BiGRU, learned queries (S0-A)',
         'C1': 'CNN-BiGRU-CTC (C1)',
         'C2': 'CNN-Transformer-CTC (C2)',
         'C3': 'CNN-Transformer decoder (C3)',
         'M0': 'MobileNetV3-Small reference (M0)'}
BLUE, GREEN, GREY, ORANGE = '#245a81', '#2e7d4f', '#6b7280', '#b8691a'
plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def validation():
    rows = data['VAL']
    names = [NAMES[r['model']] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True, sharey=True)
    for ax, values, label in [(axes[0], [r['exact'] / 10 for r in rows], 'Validation exact-sequence accuracy (%)'),
                              (axes[1], [r['cer'] * 100 for r in rows], 'Validation corpus CER (%)')]:
        ax.barh(names, values, color=BLUE); ax.invert_yaxis(); ax.set_xlabel(label)
        for i, value in enumerate(values): ax.text(value + .5, i, f'{value:.2f}', va='center')
        ax.set_xlim(0, max(values) * 1.14)
    fig.suptitle('CPU validation, 1,000 images, single training seed')
    fig.savefig(OUT / 'validation.png', dpi=150, bbox_inches='tight', pad_inches=.15); plt.close(fig)


def ablations():
    rows = data['interventions']
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    for i, row in enumerate(rows):
        ax.errorbar(row['delta'], i, xerr=[[row['delta'] - row['ci'][0]], [row['ci'][1] - row['delta']]], fmt='o', capsize=5, color=BLUE)
    ax.set_yticks(range(len(rows)), [r['name'] for r in rows])
    ax.axvline(0, color='gray', linestyle='--'); ax.invert_yaxis()
    ax.set_xlabel('Change in validation exact-sequence accuracy vs S0 (points), paired 95% interval')
    ax.set_title('Controlled interventions on S0 (augmentation intervention not run)')
    fig.savefig(OUT / 'ablations.png', dpi=150, bbox_inches='tight', pad_inches=.15); plt.close(fig)


def heldout():
    t = data['TEST']
    fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
    acc = t['exact'] / t['N'] * 100
    ax.errorbar(acc, 0, xerr=[[acc - t['wilson95'][0]], [t['wilson95'][1] - acc]], fmt='o', capsize=7, color=BLUE)
    ax.set_xlim(94, 100); ax.set_yticks([0], [NAMES['C2']])
    ax.set_xlabel('Held-out TEST exact-sequence accuracy (%), Wilson 95% interval')
    ax.set_title('One evaluation: 975 / 1,000 exact; 25 edits / 10,000 characters (CER 0.25%)')
    fig.savefig(OUT / 'heldout.png', dpi=150, bbox_inches='tight', pad_inches=.15); plt.close(fig)


def box(ax, x, y, w, h, text, fc='#eef3f8', ec=BLUE, bold=False, size=9.5, color='black'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=1.2', fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=size, color=color,
            fontweight='bold' if bold else 'normal', linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='#333', lw=1.4, shrinkA=0, shrinkB=0))


def architectures():
    fig, ax = plt.subplots(figsize=(13.7, 9.4))
    ax.set_xlim(0, 137); ax.set_ylim(0, 94); ax.axis('off')
    role = {'final': (GREEN, '#e6f3ea'), 'baseline': (ORANGE, '#fbeedd'), 'cand': (BLUE, '#eef3f8')}
    h, ys = 9.4, [78, 65, 52, 39, 26]
    rows = [
        ('CNN-BiGRU\nfixed pooling\n(S0)', 'baseline',
         ['Pool 70 tokens\nto 10 positions', '2-layer BiGRU', 'Linear: 10 x 62 logits\nargmax per position']),
        ('CNN-BiGRU\nlearned queries\n(S0-A)', 'cand',
         ['10 learned queries,\n4-head attention\nover 70 tokens', '2-layer BiGRU', 'Linear: 10 x 62 logits\nargmax per position']),
        ('CNN-BiGRU-CTC\n(C1)', 'cand',
         ['2-layer BiGRU\nover 70 tokens', 'Linear: 70 x 63 logits\n(62 characters + blank)', 'Greedy CTC decoding']),
        ('CNN-Transformer\nencoder-CTC\n(C2)', 'final',
         ['Sinusoidal positions,\n2-layer Transformer\nencoder', 'Linear: 70 x 63 logits\n(62 characters + blank)', 'Greedy CTC decoding']),
        ('CNN-Transformer\ndecoder (C3)', 'cand',
         ['Sinusoidal positions\nadded to token memory', '2-layer causal\nTransformer decoder\nwith cross-attention', 'Autoregressive decoding,\nSOS to EOS, up to 11 steps']),
    ]
    ax.text(1, 91.5, 'Shared-backbone recognizers: one CNN encoder, five different heads', fontsize=13.5, fontweight='bold', va='center')
    box(ax, 1, 54, 15, 18, 'Input\ngrayscale / 255\n1 x 90 x 282\nfull image,\nno crops,\nno resize', fc='white', ec='#333', size=9)
    box(ax, 20, 23, 20, 65, 'Shared CNN\nencoder\n\n5 conv blocks\n32, 64, 128,\n256, 256 channels\n\nwidth 282 to 70,\nheight pooled to 1\n\n70 tokens x 256', fc='#f3f0f8', ec='#5b4a8a')
    arrow(ax, 16.6, 63, 19.4, 63)
    xs = [67, 91, 115]; ws = [20, 20, 21.5]
    for (name, kind, steps), y in zip(rows, ys):
        ec, fc = role[kind]
        arrow(ax, 40.6, y + h / 2, 43.4, y + h / 2)
        box(ax, 44, y, 19, h, name, fc=fc, ec=ec, bold=True, size=9)
        arrow(ax, 63.6, y + h / 2, 66.4, y + h / 2)
        for k, (x, w, t) in enumerate(zip(xs, ws, steps)):
            box(ax, x, y, w, h, t, size=8.6)
            if k < 2: arrow(ax, x + w + .6, y + h / 2, xs[k + 1] - .6, y + h / 2)
    ax.plot([1, 136], [20, 20], color='#bbb', lw=1, ls='--')
    ax.text(1, 17.5, 'Separate reference: its own backbone, not the shared CNN', fontsize=11.5, fontweight='bold', va='center')
    y = 5
    box(ax, 1, y, 15, h, 'Input\nsame image', fc='white', ec='#333', size=9)
    arrow(ax, 16.6, y + h / 2, 19.4, y + h / 2)
    box(ax, 20, y, 20, h, 'Replicate to 3\nchannels, ImageNet\nnormalization', size=8.6)
    arrow(ax, 40.6, y + h / 2, 43.4, y + h / 2)
    box(ax, 44, y, 19, h, 'MobileNetV3-Small\nreference (M0)', fc=role['cand'][1], ec=role['cand'][0], bold=True, size=9)
    arrow(ax, 63.6, y + h / 2, 66.4, y + h / 2)
    box(ax, 67, y, 20, h, 'MobileNetV3-Small\nfeatures, random init', size=8.6)
    arrow(ax, 87.6, y + h / 2, 90.4, y + h / 2)
    box(ax, 91, y, 20, h, 'Pool to 10 positions', size=8.6)
    arrow(ax, 111.6, y + h / 2, 114.4, y + h / 2)
    box(ax, 115, y, 21.5, h, 'Linear: 10 x 62 logits\nargmax per position', size=8.6)
    handles = [Patch(fc=role['final'][1], ec=role['final'][0], label='Final model (reported TEST result)'),
               Patch(fc=role['baseline'][1], ec=role['baseline'][0], label='Baseline'),
               Patch(fc=role['cand'][1], ec=role['cand'][0], label='Compared on validation')]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.03), ncol=3, frameon=False, fontsize=10)
    fig.savefig(OUT / 'architectures.png', dpi=100, bbox_inches='tight', pad_inches=.2); plt.close(fig)


if __name__ == '__main__':
    validation(); ablations(); heldout(); architectures()
