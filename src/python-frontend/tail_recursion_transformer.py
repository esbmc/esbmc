"""
Tail Recursion to Loop Transformation for ESBMC Python Frontend.
This module detects tail recursive functions and converts them to iterative loops,
improving verification performance by reducing unwinding depth in bounded model checking.
"""

import ast
from typing import List
from dataclasses import dataclass

@dataclass
class TailCall:
    name: str
    args: List[ast.expr]
    lineno: int


class TailCallDetector(ast.NodeVisitor):
    """Detect if function is tail recursive"""
    def __init__(self, fname: str):
        # Store the target function name to detect
        self.fname = fname

        # Assume it's tail recursive (set to False if blocking patterns found)
        self.is_tail = True

        # List to collect all detected tail calls
        self.calls: List[TailCall] = []

        # Flag to track if we're currently inside a return statement
        self.in_return = False

    def visit_Return(self, n: ast.Return) -> None:
        # Check if this is an empty return (returns None)
        if n.value is None:
            # Empty return breaks tail recursion pattern
            self.is_tail = False
            return
        
        # Mark that we're entering a return statement context
        self.in_return = True

        # Check if this return contains a call to our target function
        # Three conditions must all be true:
        # 1. Return value is a Call (function call)
        # 2. Call is to a Name (not a method or other expression)
        # 3. The name matches our target function (self.fname)
        if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == self.fname:
            # Found a tail recursive call! Record it for later use
            self.calls.append(TailCall(self.fname, n.value.args, n.lineno))
        
        # Clear the return statement context flag
        self.in_return = False

    def visit_Call(self, n: ast.Call) -> None:
        # Check if this is a recursive call in an INVALID position
        # Valid position: inside a return statement (in_return = True)
        # Invalid position: anywhere else (in_return = False)
        if not self.in_return and isinstance(n.func, ast.Name) and n.func.id == self.fname:
            # Found a recursive call that's NOT in a return statement
            # This breaks the tail recursion pattern
            self.is_tail = False

        # Continue visiting child nodes (arguments, nested calls, etc.)
        self.generic_visit(n)

    def visit_FunctionDef(self, n: ast.FunctionDef) -> None:
        # Only process the target function, skip all others
        # This prevents accidental analysis of nested functions
        if n.name == self.fname:
            # This is our target function, visit its body
            self.generic_visit(n)
        # If n.name != self.fname: silently skip this function


class TailCallTransformer(ast.NodeTransformer):
    """Transform tail recursive function to iterative loop"""
    def __init__(self, fname: str, params: List[str]):
        # Store the function name to transform
        self.fname = fname

        # Store the list of function parameters (will be reassigned in the loop)
        # Example: ["n", "acc"] for def func(n, acc):
        self.params = params
    
    def visit_FunctionDef(self, n: ast.FunctionDef) -> ast.FunctionDef:
        # Only transform if this is the target function
        if n.name != self.fname:
            return n
        
        # Transform the function: replace recursive calls with loop
        # Create new FunctionDef with:
        # - Same name, args, decorators, return type
        # - NEW body: while True loop containing transformed statements
        return ast.FunctionDef(
            name=n.name, args=n.args,
            # Wrap function body in "while True:" loop
            body=[ast.While(
                test=ast.Constant(value=True), # while True:
                body=self._transform(n.body),  # transformed statements
                orelse=[]                      # no else clause
            )],
            decorator_list=n.decorator_list, returns=n.returns,
            lineno=n.lineno, col_offset=n.col_offset
        )

    def _transform(self, stmts: List[ast.stmt]) -> List[ast.stmt]:
         # Transform all statements in the function body
        # Converts "return func(...)" to parameter assignments + continue
        result = []
        for stmt in stmts:
            # Check if this statement is a tail recursive call: "return func(...)"
            if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                call = stmt.value

                # Check if this call is to our target function
                if isinstance(call.func, ast.Name) and call.func.id == self.fname:
                    # Found a tail call! Convert it to assignments + continue
                    result.extend(self._tail_to_loop(call))
                    continue # Skip the original return statement
            
            # Handle if/elif/else statements (need to transform their bodies too)
            if isinstance(stmt, ast.If):
                result.append(self._transform_if(stmt))
            else:
                # Keep other statements unchanged (visits them normally)
                result.append(self.visit(stmt))
        return result


    def _transform_if(self, n: ast.If) -> ast.If:
        # Recursively transform if/elif/else statements
        # Needed because tail calls might be inside if branches

        orelse = []
        if n.orelse:
            # Check if else clause is another if (elif)
            orelse = ([self._transform_if(n.orelse[0])] if isinstance(n.orelse[0], ast.If) 
                     else self._transform(n.orelse))
        
        # Return new If node with transformed body and orelse
        return ast.If(test=n.test, body=self._transform(n.body), orelse=orelse,
                     lineno=n.lineno, col_offset=n.col_offset)


    def _tail_to_loop(self, call: ast.Call) -> List[ast.stmt]:
        # Convert tail recursive call to parameter assignments + continue
        # Example: return func(n-1, n*acc) becomes:
        #   n: int = n-1
        #   acc: int = n*acc
        #   continue

        stmts = []

        # Check if all arguments are "simple" (Name, Constant, or unary minus)
        # Simple args can be directly assigned: n = x
        # Complex args need temporary variables to preserve evaluation order
        is_simple = all(isinstance(a, (ast.Name, ast.Constant)) or 
                       (isinstance(a, ast.UnaryOp) and isinstance(a.op, ast.USub))
                       for a in call.args)

        if is_simple:
            # SIMPLE CASE: All arguments are simple
            # Direct assignment to parameters with type annotation
            # Example: return func(x, y) → x: Type = x; y: Type = y
            for i, arg in enumerate(call.args):
                if i < len(self.params):
                    stmts.append(ast.AnnAssign(
                        target=ast.Name(id=self.params[i], ctx=ast.Store()),
                        annotation=self._type(arg),
                        value=arg, simple=1,
                        lineno=call.lineno, col_offset=call.col_offset
                    ))
        else:
            # COMPLEX CASE: Arguments contain expressions (BinOp, Subscript, etc)
            # Need temporary variables to preserve evaluation order
            # Example: return func(n-1, n*acc) evaluates with ORIGINAL n

            temps = []

            # STEP 1: Evaluate all arguments into temporary variables
            # This ensures evaluation happens with original parameter values
            for i, arg in enumerate(call.args):
                t = f"_t{i}"  # Temporary variable name: _t0, _t1, etc.
                stmts.append(ast.AnnAssign(
                    target=ast.Name(id=t, ctx=ast.Store()),
                    annotation=self._type(arg),
                    value=arg, simple=1,
                    lineno=call.lineno, col_offset=call.col_offset
                ))
                temps.append(t)

            # STEP 2: Assign temporary values to actual parameters
            # Now that all expressions are evaluated, assign to parameters
            for i, t in enumerate(temps):
                if i < len(self.params):
                    stmts.append(ast.Assign(
                        targets=[ast.Name(id=self.params[i], ctx=ast.Store())],
                        value=ast.Name(id=t, ctx=ast.Load()),
                        lineno=call.lineno, col_offset=call.col_offset
                    ))

        # Add continue statement to jump back to while loop
        # This replaces the return and starts next iteration
        stmts.append(ast.Continue(lineno=call.lineno, col_offset=call.col_offset))
        return stmts

    # # TODO: Add a type-inference module to the preprocessing stage.
    @staticmethod
    def _type(n: ast.expr) -> ast.expr:
        # Infer type annotation from expressions
        # Returns: ast.Name with type (int, float, str, bool, Any)

        # Check if expression is a binary operation (arithmetic)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv)):
            return ast.Name(id='int', ctx=ast.Load())
        if isinstance(n, ast.Constant):
            if isinstance(n.value, int): return ast.Name(id='int', ctx=ast.Load())
            if isinstance(n.value, float): return ast.Name(id='float', ctx=ast.Load())
            if isinstance(n.value, str): return ast.Name(id='str', ctx=ast.Load())
            if isinstance(n.value, bool): return ast.Name(id='bool', ctx=ast.Load())
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return TailCallTransformer._type(n.operand)
        return ast.Name(id='Any', ctx=ast.Load())


def transform_tail_recursion(code: str) -> str:
    """Transform tail recursive functions to iterative loops"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Find tail recursive functions
    to_transform = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            det = TailCallDetector(node.name)
            for stmt in node.body:
                det.visit(stmt)
            if det.is_tail and det.calls:
                params = [arg.arg for arg in node.args.args]
                to_transform[node.name] = params

    if not to_transform:
        return ast.unparse(tree)

    # Transform tree
    class TreeTrans(ast.NodeTransformer):
        def visit_FunctionDef(self, n):
            self.generic_visit(n)
            return TailCallTransformer(n.name, to_transform[n.name]).visit(n) if n.name in to_transform else n

    return ast.unparse(TreeTrans().visit(tree))


if __name__ == "__main__":
    code = '''
def fib(n, a=0, b=1):
    if n == 0:
        return a
    return fib(n - 1, b, a + b)
'''
    print("BEFORE:\n", code)
    print("AFTER:\n", transform_tail_recursion(code))