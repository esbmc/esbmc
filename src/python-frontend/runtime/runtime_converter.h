#pragma once

#include <nlohmann/json.hpp>
#include <util/irep/std_code.h>
#include <util/irep/std_types.h>
#include <util/symtab/context.h>

#include <map>
#include <set>
#include <string>
#include <vector>

/// Lowers a Python module onto the pyrt runtime models
/// (src/c2goto/library/python/runtime) for --python-runtime. Every value is a
/// `PyRtObject *` and every operation a call into the runtime, so dispatch on
/// the object's type is left to symbolic execution.
class python_runtime_converter
{
public:
  python_runtime_converter(contextt &context, const nlohmann::json &ast);

  void convert();

private:
  using json = nlohmann::json;

  contextt &context_;
  const json &ast_;
  const std::string file_;
  const typet object_type_;

  std::string function_;
  std::set<std::string> locals_;
  std::set<std::string> globals_;
  std::map<std::string, const json *> functions_;
  code_blockt *block_ = nullptr;
  unsigned temporaries_ = 0;

  [[noreturn]] void unsupported(const json &node) const;
  locationt location(const json &node) const;
  const symbolt &lookup(const std::string &id) const;
  symbolt &add_symbol(symbolt &symbol);

  std::string global_id(const std::string &name) const;
  std::string function_id(const std::string &name) const;
  std::string
  local_id(const std::string &function, const std::string &name) const;

  exprt runtime_object(const std::string &name) const;
  exprt new_temporary(const typet &type, const locationt &loc);
  exprt call(
    const std::string &function,
    const std::vector<exprt> &arguments,
    const locationt &loc);
  void raise(const std::string &message, const locationt &loc);
  void emit_if(const exprt &cond, codet then_case, const locationt &loc);

  exprt expr(const json &node);
  exprt truth(const json &node);
  exprt constant(const json &node);
  exprt int_constant(int64_t value, const locationt &loc);
  exprt name(const json &node);
  exprt binop(const json &node);
  exprt unaryop(const json &node);
  exprt compare(const json &node);
  exprt boolop(const json &node);
  void boolop_rest(
    const json &values,
    size_t index,
    const exprt &result,
    bool is_and,
    const locationt &loc);
  exprt call_expr(const json &node);
  exprt list(const json &node);
  exprt subscript(const json &node);

  void statements(const json &body, code_blockt &block);
  void statement(const json &node);
  void store(const json &target, const exprt &value, const locationt &loc);
  void if_statement(const json &node);
  void while_statement(const json &node);
  void assert_statement(const json &node);

  void collect_assigned(
    const json &body,
    std::set<std::string> &assigned,
    std::set<std::string> &declared_global) const;
  void declare_variable(
    const std::string &id,
    const std::string &name,
    bool is_global,
    const locationt &loc);
  void declare_function(const json &def);
  void define_function(const json &def);
  void add_c_intrinsics();
  void add_entry_points(code_blockt &user_code);
};
