#pragma once

#include <ld-frontend/parser/ld_ast.h>
#include <stdexcept>
#include <string>
#include <unordered_map>

struct TypeCheckError : std::runtime_error
{
  explicit TypeCheckError(const std::string &msg) : std::runtime_error(msg)
  {
  }
};

class TypeChecker
{
public:
  // Validate the AST and annotate variable types.
  // Throws TypeCheckError on any violation.
  void check(const LdAst &ast);

private:
  std::unordered_map<std::string, VarKind> var_types_;

  void build_var_type_map(const LdAst &ast);
  void check_rung_element(const RungElement &elem);
  void check_timer_fb(const TimerFBNode &fb);
  void check_counter_fb(const CounterFBNode &fb);
  void check_arith_fb(const ArithFBNode &fb);

  VarKind lookup_type(const std::string &var, const LdLocation &loc) const;

  static std::string loc_str(const LdLocation &loc);
};
