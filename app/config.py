import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Points at the real transactional export. Override with the DATA_PATH env
# var to point at a different file without editing this, e.g.:
#   export DATA_PATH=/path/to/other_data.csv
# data/placeholder_data.csv (synthetic, same schema) is kept around for
# testing the pipeline without the real data.
DEFAULT_DATA_PATH = BASE_DIR / "data" / "data.csv"
DATA_PATH = Path(os.environ.get("DATA_PATH", str(DEFAULT_DATA_PATH)))

RANDOM_STATE = 0
ISOLATION_FOREST_CONTAMINATION = 0.05
PCA_N_COMPONENTS = 6
N_CLUSTERS = 3
TOP_N_PRODUCTS_PER_CLUSTER = 10
TOP_N_RECOMMENDATIONS = 3
