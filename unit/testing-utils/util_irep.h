// Helper to generate types, expressions
#pragma once
#include <util/expr.h>
#include <util/std_types.h>

enum class Builtin_Type
{
  Void,
  Bool,
  UChar,
  WChar_U,
  Char16,
  Char32,
  UShort,
  UInt,
  ULong,
  ULongLong,
  SChar,
  WChar_S,
  Short,
  Int,
  Long,
  LongLong,
  Half,
  Float,
  Double,
  LongDouble,
  Last // Used for castings
};

void gen_builtin_type(typet &new_type, Builtin_Type bt);

struct_union_typet::componentt gen_component(const char *name, Builtin_Type bt);

union_typet gen_union(std::vector<struct_union_typet::componentt> &v);