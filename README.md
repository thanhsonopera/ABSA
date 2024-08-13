# ABSA

hello world 

## Prerequisites

- Python 3.11.5
- Required packages:
  - `ydata-profiling`
  - `torch`
  ```bash
  pip install -r requirements.txt
  ```
## Usage

### 1. Convert XML to CSV

```bash
python semeval/processing/xlmtocsv.py --dm <domain> --lang <language> --tp <task_type>
```
### 2. Convert CSV to Aspect
```bash
python python semeval/processing/datatoaspect.py --dm <domain> --lang <language> --tp <task_type>
```
### 3. Merge Aspect
```bash
python semeval/processing/relabel.py --dm <domain> --lang <language> --tp <task_type> [--ck True]
```
