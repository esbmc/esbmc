#pragma once

enum class StatementType
{
  VARIABLE_ASSIGN,
  COMPOUND_ASSIGN,
  FUNC_DEFINITION,
  IF_STATEMENT,
  WHILE_STATEMENT,
  UNKNOWN,
};

enum class ExpressionType
{
  UNARY_OPERATION,
  BINARY_OPERATION,
  LITERAL,
  VARIABLE_REF,
  UNKNOWN,
};

static const std::unordered_map<std::string, std::string> operator_map = {
  {"Add", "+"},
  {"Sub", "-"},
  {"Mult", "*"},
  {"Div", "/"},
  {"BitOr", "bitor"},
  {"BitAnd", "bitand"},
  {"BitXor", "bitxor"},
  {"Invert", "bitnot"},
  {"LShift", "shl"},
  {"RShift", "ashr"},
  {"USub", "unary-"},
  {"Eq", "="},
  {"Lt", "<"},
  {"Gt", ">"},
};

static const std::unordered_map<std::string, StatementType> statement_map = {
  {"AnnAssign", StatementType::VARIABLE_ASSIGN},
  {"Assign", StatementType::VARIABLE_ASSIGN},
  {"FunctionDef", StatementType::FUNC_DEFINITION},
  {"If", StatementType::IF_STATEMENT},
  {"AugAssign", StatementType::COMPOUND_ASSIGN},
  {"While", StatementType::WHILE_STATEMENT},
};
