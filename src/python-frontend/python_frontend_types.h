#pragma once

#include <string>

enum class StatementType
{
  VARIABLE_ASSIGN,
  COMPOUND_ASSIGN,
  FUNC_DEFINITION,
  IF_STATEMENT,
  WHILE_STATEMENT,
  EXPR,
  RETURN,
  ASSERT,
  CLASS_DEFINITION,
  PASS,
  IMPORT,
  UNKNOWN,
};

enum class ExpressionType
{
  UNARY_OPERATION,
  BINARY_OPERATION,
  LOGICAL_OPERATION,
  LITERAL,
  IF_EXPR,
  VARIABLE_REF,
  FUNC_CALL,
  UNKNOWN,
};

bool is_builtin_type(const std::string &name);
