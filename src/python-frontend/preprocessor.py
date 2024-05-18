import ast

class Preprocessor(ast.NodeTransformer):
    def __init__(self):
        self.target_name = ""

    def visit_For(self, node):
        # Transformation from for to while if the iterator is range
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            start = node.iter.args[0]
            end = node.iter.args[1]
            if len(node.iter.args) > 2:
                step = node.iter.args[2]
            else:
                step = ast.Constant(value=1)

            start_assign = ast.AnnAssign(target=ast.Name(id='start', ctx=ast.Store()), annotation=ast.Name(id='int', ctx=ast.Load()), value=start, simple=1)
            has_next_call = ast.Call(func=ast.Name(id='ESBMC_range_has_next_', ctx=ast.Load()), args=[start, end, step], keywords=[])
            has_next_assign = ast.AnnAssign(target=ast.Name(id='has_next', ctx=ast.Store()), annotation=ast.Name(id='bool', ctx=ast.Load()), value=has_next_call, simple=1)
            has_next_name = ast.Name(id='has_next', ctx=ast.Load())
            while_cond = ast.Compare(left=has_next_name, ops=[ast.Eq()], comparators=[ast.Constant(value=True)])
            transformed_body = []
            self.target_name = node.target.id
            for statement in node.body:
                transformed_body.append(self.visit(statement))
            while_body = transformed_body + [ast.Assign(targets=[ast.Name(id='start', ctx=ast.Store())], value=ast.Call(func=ast.Name(id='ESBMC_range_next_', ctx=ast.Load()), args=[ast.Name(id='start', ctx=ast.Load()), step], keywords=[])), ast.Assign(targets=[has_next_name], value=ast.Call(func=ast.Name(id='ESBMC_range_has_next_', ctx=ast.Load()), args=[ast.Name(id='start', ctx=ast.Load()), end, step], keywords=[]))]
            while_stmt = ast.While(test=while_cond, body=while_body, orelse=[])
            return [start_assign, has_next_assign, while_stmt]
        return node

    def visit_Name(self, node):
        # Replace variable names as needed in the for to while transformation
        if node.id == self.target_name:
            node.id = 'start'
        return node

    def visit_Call(self, node):
        # Transformation for int.from_bytes
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "int" and node.func.attr == "from_bytes":
            if len(node.args) > 1 and isinstance(node.args[1], ast.Str) and node.args[1].s == 'big':
                node.args[1] = ast.NameConstant(value=True)
            else:
                node.args[1] = ast.NameConstant(value=False)
        self.generic_visit(node)
        return node