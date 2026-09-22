import pyarrow.parquet as pq
import glob

def clean_parquets():
    out_dir = "/csehome/m25csa007/m25csa007/rjm-ltr/data/processed"
    for file in glob.glob(f"{out_dir}/*.parquet"):
        try:
            # Read metadata and dataset
            table = pq.read_table(file)
            
            # Find columns to keep (first occurrence of each column name)
            seen = set()
            keep_indices = []
            for i, name in enumerate(table.column_names):
                if name not in seen and name != "__index_level_0__":
                    seen.add(name)
                    keep_indices.append(i)
            
            # Select columns
            clean_table = table.select(keep_indices)
            
            # Write back
            pq.write_table(clean_table, file)
            print(f"Cleaned {file}")
        except Exception as e:
            print(f"Error cleaning {file}: {e}")

if __name__ == "__main__":
    clean_parquets()
