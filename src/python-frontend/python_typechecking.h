#pragma once

#include <python-frontend/symbol_id.h>

#include <nlohmann/json.hpp>
#include <util/expr.h>
#include <util/std_code.h>
#include <util/std_types.h>
#include <util/type.h>
#include <util/symbol.h>
#include <util/python_types.h>

#include <string>
#include <unordered_map>
#include <vector>

class python_converter;

class python_typechecking
{
public:
  explicit python_typechecking(python_converter &converter);

  std::vector<typet>
  collect_annotation_types(const nlohmann::json &annotation) const;

  void cache_annotation_types(
    symbolt &symbol,
    const nlohmann::json &annotation);

  std::vector<typet> get_annotation_types(const std::string &symbol_id) const;

  void inject_parameter_type_assertions(
    const nlohmann::json &function_node,
    const symbol_id &function_id,
    const code_typet &type,
    exprt &function_body);

  bool should_skip_type_assertion(const typet &annotated_type) const;

  exprt build_isinstance_check(
    const exprt &value_expr,
    const typet &annotated_type) const;

  bool build_type_assertion(
    const exprt &value_expr,
    const typet &annotated_type,
    const std::vector<typet> &allowed_types,
    const std::string &context_name,
    const locationt &location,
    code_assertt &out_assert) const;

  void emit_type_annotation_assertion(
    const exprt &value_expr,
    const typet &annotated_type,
    const std::vector<typet> &allowed_types,
    const std::string &context_name,
    const locationt &location,
    codet &target_block);

  std::string get_constructor_name(const nlohmann::json &func_node) const;

  bool class_derives_from(
    const std::string &class_name,
    const std::string &expected_base) const;

private:
  python_converter &converter_;
  std::unordered_map<std::string, std::vector<typet>> annotation_type_cache_;
};
