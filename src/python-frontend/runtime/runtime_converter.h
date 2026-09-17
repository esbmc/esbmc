#pragma once

#include <nlohmann/json.hpp>
#include <util/irep/std_code.h>
#include <util/irep/std_types.h>
#include <util/symtab/context.h>

#include <functional>
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
  python_runtime_converter(
    contextt &context,
    const nlohmann::json &ast,
    bool check_annotations = false);

  void convert();

private:
  using json = nlohmann::json;

  contextt &context_;
  const json &ast_;
  const std::string file_;
  const typet object_type_;
  const bool check_annotations_;

  /// Return annotation of the function being converted, or null.
  const nlohmann::json *return_annotation_ = nullptr;

  /// Annotated names of the scope being converted, and their annotations.
  std::map<std::string, const nlohmann::json *> annotated_;

  /// Non-zero while converting a try body, where an error a model records can
  /// become a Python exception instead of an abort.
  unsigned in_try_ = 0;

  /// Symbol id of the function being converted; empty at module level.
  std::string code_id_;
  std::string function_name_;
  std::set<std::string> locals_;
  std::set<std::string> globals_;
  std::map<std::string, const json *> functions_;
  std::map<std::string, const json *> classes_;
  std::map<std::string, std::string> string_literals_;
  std::map<double, std::string> float_literals_;
  code_blockt *block_ = nullptr;
  unsigned temporaries_ = 0;

  [[noreturn]] void unsupported(const json &node) const;
  locationt location(const json &node) const;
  const symbolt &lookup(const std::string &id) const;
  symbolt &add_symbol(symbolt &symbol);

  std::string global_id(const std::string &name) const;
  std::string code_id(const std::string &cls, const std::string &name) const;
  std::string local_id(const std::string &name) const;
  std::string function_object_id(const std::string &cls, const std::string &name)
    const;
  std::string type_object_id(const std::string &cls) const;

  exprt address(const std::string &id) const;
  exprt type_pointer(const std::string &cls) const;
  exprt name_pointer(const std::string &name);
  exprt struct_value(const char *tag, const std::map<std::string, exprt> &fields)
    const;
  void add_static_object(
    const std::string &id,
    const char *tag,
    const locationt &loc);

  exprt new_temporary(const typet &type, const locationt &loc);
  exprt call(
    const std::string &function,
    const std::vector<exprt> &arguments,
    const locationt &loc);
  std::vector<exprt> arguments(const json &call_node, bool allow_keywords = false);
  std::vector<exprt> arguments_for(
    const json &call_node,
    const json &signature,
    const std::string &callee,
    const locationt &loc);
  exprt arguments_struct(
    const std::vector<exprt> &arguments,
    const json &call_node) const;
  void raise(const std::string &message, const locationt &loc);
  void emit_if(const exprt &cond, codet then_case, const locationt &loc);

  exprt expr(const json &node);
  exprt truth(const json &node);
  exprt constant(const json &node);
  exprt int_constant(int64_t value, const locationt &loc);
  exprt str_constant(const std::string &value, const locationt &loc);
  exprt float_constant(double value, const locationt &loc);
  exprt name(const json &node);
  exprt binop(const std::string &op, exprt left, exprt right, const json &node);
  exprt unaryop(const json &node);
  exprt compare(const json &node);
  exprt boolop(const json &node);
  void boolop_rest(
    const json &values,
    size_t index,
    const exprt &result,
    bool is_and,
    const locationt &loc);
  exprt ifexp(const json &node);
  exprt annotation_type(const json &annotation) const;
  void collect_annotations(const json &body);
  void check_annotation(
    const json &annotation,
    const exprt &value,
    const std::string &what,
    const locationt &loc);
  exprt call_expr(const json &node);
  exprt list(const json &node);
  exprt tuple(const json &node);
  exprt comprehension(const json &node, bool is_dict);
  void emit_loop(
    const json &target,
    const json &iterable,
    const std::function<void()> &emit_body,
    const locationt &loc);
  exprt dict_literal(const json &node);
  exprt subscript(const json &node);

  void statements(const json &body, code_blockt &block);
  void statement(const json &node);
  void store(const json &target, const exprt &value, const locationt &loc);
  void aug_assign(const json &node);
  void if_statement(const json &node);
  void while_statement(const json &node);
  void for_statement(const json &node);
  void assert_statement(const json &node);
  void raise_statement(const json &node);
  void try_statement(const json &node);
  void emit_guarded(const json &node, const locationt &loc);
  void throw_pending(const locationt &loc);
  void class_statement(const json &node);

  void collect_assigned(
    const json &body,
    std::set<std::string> &assigned,
    std::set<std::string> &declared_global) const;
  void declare_variable(
    const std::string &id,
    const std::string &name,
    const typet &type,
    bool is_global,
    const locationt &loc);
  void check_signature(const json &def) const;
  void declare_function(const json &def, const std::string &cls);
  void define_function(const json &def, const std::string &cls);
  void declare_class(const json &def);
  void add_c_intrinsics();
  void add_entry_points(code_blockt &user_code);
};
