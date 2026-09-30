import ast
from annotate_ast import type_to_ast_annotation
from pytype_infer.lattice import (
    IntType,
    FloatType,
    BoolType,
    StrType,
    ListType,
    UnionType,
)


def test1():
    d = {}
    d.setdefault("x",1)
    return d

def test2():
    d = {}
    x = d.setdefault("x", 1)
    return x

def test3():
  d ={"x": 1}
  x = d.setdefault("x", 2.0)
  return x

def test4():
   d = {}
   x = d.setdefault("x")
   return x

assert ast.unparse(type_to_ast_annotation(ListType(IntType()))) == "list[int]"

assert ast.unparse(
    type_to_ast_annotation(UnionType([IntType(), StrType()]))
) == "int | str"