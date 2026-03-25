

def load_schema():
    with open(r'/mlflow/model_artifacts/1/models/m-c9b8782d85e94044a200b6e9095e3238/artifacts/schemas.json', 'r') as f:
        schemas = json.load(f)
    return schemas

def find_dtype(dtype: str):
    if 'int' in dtype:
        return int
    elif 'float' in dtype:
        return float
    else:
        return str

def generate_valid_payload():
    schema = load_schema()
    data = {}
    for col, meta in schema.items():
        dtype = find_dtype(meta["dtype"])
        
        if 'allowed_values' in meta and meta['allowed_values']:
            data[col] = meta['allowed_values'][0]
        elif dtype in [int, float]:
            data[col] = int(meta['min']+meta['max'])/2
        else:
            data[col] = "Testing the ML API"
    return data

