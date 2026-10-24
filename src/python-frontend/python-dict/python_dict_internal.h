#pragma once

// Shared include bundle for the python-dict module translation units.  This is
// the exact set of headers the original monolithic python_dict_handler.cpp
// pulled in, so every split unit sees the same declarations it did before the
// refactor.
#include <python-frontend/json_utils.h>
#include <python-frontend/python-dict/python_dict_handler.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/python-list/python_list.h>
#include <python-frontend/string/string_builder.h>
#include <python-frontend/tuple/tuple_handler.h>
#include <python-frontend/type/type_handler.h>
#include <irep2/irep2_utils.h>
#include <util/arith/arith_tools.h>
#include <util/lang/c_types.h>
#include <util/config/config.h>
#include <util/symtab/context.h>
#include <util/irep/migrate.h>
#include <util/lang/python_types.h>
#include <util/irep/std_code.h>

#include <algorithm>
#include <functional>
#include <sstream>

/// The Constant that statement @p node binds to @p name alone, as in
/// `name = c` or `name: T = c`, else null.
inline const nlohmann::json *
constant_bound_to(const nlohmann::json &node, const std::string &name)
{
  const std::string type = node.value("_type", "");
  const nlohmann::json *target = type == "Assign" && node["targets"].size() == 1
                                   ? &node["targets"][0]
                                 : type == "AnnAssign" ? &node["target"]
                                                       : nullptr;
  if (
    !target || target->value("_type", "") != "Name" ||
    target->value("id", "") != name || !node["value"].is_object() ||
    node["value"].value("_type", "") != "Constant")
    return nullptr;
  return &node["value"];
}

/// Counts every binding of @p name in @p node (a Name it stores or deletes, a
/// parameter, def, import, handler or global declaration of that name), and
/// sets @p value to the Constant of a lone `name = c` or `name: T = c`. Any
/// other occurrence of the name as a string, e.g. an attribute, also counts:
/// a false binding only loses the resolution.
inline void count_name_bindings(
  const nlohmann::json &node,
  const std::string &name,
  unsigned &count,
  const nlohmann::json *&value)
{
  if (node.is_string())
    count += node.get<std::string>() == name;
  if (node.is_array())
    for (const auto &item : node)
      count_name_bindings(item, name, count, value);
  if (!node.is_object())
    return;
  const std::string type = node.value("_type", "");
  if (type == "Name" || type == "Constant")
  {
    const bool load =
      node.contains("ctx") && node["ctx"].value("_type", "") == "Load";
    count += type == "Name" && node.value("id", "") == name && !load;
    return;
  }
  if (const nlohmann::json *constant = constant_bound_to(node, name))
    value = constant;
  for (const auto &item : node.items())
    if (item.key() != "_type")
      count_name_bindings(item.value(), name, count, value);
}

/// The source literal of a constant dict key, e.g. `1` or `"a"`, or of the
/// constant a key name is bound to when @p ast binds that name nowhere else,
/// in any scope; empty for any other key. A value's recorded type is filed
/// under it, so a read at a constant key consults only the values written
/// there.
inline std::string
dict_key_id(const nlohmann::json &key, const nlohmann::json &ast)
{
  if (!key.is_object())
    return std::string();
  const std::string type = key.value("_type", "");
  if (type == "Constant")
    return key["value"].dump();
  if (type != "Name")
    return std::string();
  unsigned count = 0;
  const nlohmann::json *value = nullptr;
  count_name_bindings(ast, key.value("id", ""), count, value);
  return count == 1 && value ? (*value)["value"].dump() : std::string();
}
