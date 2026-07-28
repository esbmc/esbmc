import ast, json
def read_source_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
