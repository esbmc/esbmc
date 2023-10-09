import ast
import sys
from ast2json import ast2json
import json

def main():
    if len(sys.argv) != 2:
        print("Usage: python astgen.py <file path>")
        return

    filename = sys.argv[1]

    with open(filename, "r") as source:
        tree = ast.parse(source.read())

    ast_json = ast2json(tree)
    json_filename = "/tmp/ast.json"

    with open(json_filename, "w") as json_file:
        json.dump(ast_json, json_file, indent=4)

if __name__ == "__main__":
    main()
