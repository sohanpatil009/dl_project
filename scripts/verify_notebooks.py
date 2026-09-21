import json
from pathlib import Path

for nb in ['notebooks/wideband_signal_recognition_colab.ipynb',
           'notebooks/wideband_signal_recognition_kaggle.ipynb',
           'notebooks/wideband_signal_recognition_jupyter.ipynb']:
    d = json.loads(Path(nb).read_text(encoding='utf-8'))
    assert d['nbformat'] == 4, nb
    n_md = sum(1 for c in d['cells'] if c['cell_type'] == 'markdown')
    n_code = sum(1 for c in d['cells'] if c['cell_type'] == 'code')
    print(nb + ': %d cells (%d md, %d code) OK' % (len(d['cells']), n_md, n_code))
    bad = 0
    for i, c in enumerate(d['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        lines = [l for l in src.splitlines()
                 if not l.strip().startswith(('!', '%'))]
        try:
            compile('\n'.join(lines), '%s#%d' % (nb, i), 'exec')
        except SyntaxError as e:
            bad += 1
            print('  SYNTAX ERROR cell %d: %s' % (i, e))
    print('  syntax errors: %d' % bad)

# Check kaggle paths
d = json.loads(Path('notebooks/wideband_signal_recognition_kaggle.ipynb').read_text(encoding='utf-8'))
full = '\n'.join(''.join(c['source']) for c in d['cells'])
assert '/kaggle/input/' in full, 'missing /kaggle/input'
assert '/kaggle/working/' in full, 'missing /kaggle/working'
assert '/content/drive' not in full, 'kaggle nb must not reference drive'
print('Kaggle path checks OK')

d = json.loads(Path('notebooks/wideband_signal_recognition_colab.ipynb').read_text(encoding='utf-8'))
heads = [ ''.join(c['source'])[:80] for c in d['cells'] if c['cell_type'] == 'markdown']
print('Colab sections:', len(heads))

# Local-Jupyter notebook must not reference cloud paths
d = json.loads(Path('notebooks/wideband_signal_recognition_jupyter.ipynb').read_text(encoding='utf-8'))
full = '\n'.join(''.join(c['source']) for c in d['cells'])
assert '/content/' not in full, 'jupyter nb must not reference /content'
assert '/kaggle/' not in full, 'jupyter nb must not reference /kaggle'
assert 'google.colab' not in full, 'jupyter nb must not reference google.colab'
assert 'DATA_ROOT = ' in full, 'jupyter nb should use relative DATA_ROOT'
print('Jupyter local-path checks OK')
