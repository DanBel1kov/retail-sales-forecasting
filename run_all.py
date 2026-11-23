from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parent
for name in ['01_analysis.ipynb', '02_forecasting.ipynb']:
    print(f'Выполняется {name}', flush=True)
    notebook = nbformat.read(root / name, as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                            resources={'metadata': {'path': str(root)}})
    client.execute()
    nbformat.write(notebook, root / name)
    print(f'Готово: {name}', flush=True)
print('Оба ноутбука выполнены без ошибок.', flush=True)
