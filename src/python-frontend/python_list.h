#pragma once

#include <nlohmann/json.hpp>
#include <util/type.h>
#include <set>
#include <utility>

class exprt;
class symbolt;
class python_converter;

using TypeInfo = std::vector<std::pair<std::string, typet>>;

class python_list
{
public:
  python_list(python_converter &converter, const nlohmann::json &list)
    : converter_(converter), list_value_(list)
  {
  }

  exprt get();

  exprt index(const exprt &array, const nlohmann::json &slice_node);

  exprt compare(const exprt &l1, const exprt &l2, const std::string &op);

  exprt list_repetition(
    const nlohmann::json &left_node,
    const nlohmann::json &right_node,
    const exprt &lhs,
    const exprt &rhs);

  exprt build_push_list_call(
    const symbolt &list,
    const nlohmann::json &op,
    const exprt &elem);

  void add_type_info(
    const std::string &list_symbol_id,
    const std::string &elem_id,
    const typet &elem_type)
  {
    list_type_map[list_symbol_id].push_back(std::make_pair(elem_id, elem_type));
  }

  static void
  copy_type_info(const std::string &source_list, const std::string dest_list)
  {
    if (!list_type_map[source_list].empty())
    {
      list_type_map[dest_list] = list_type_map[source_list];
    }
  }

private:
  exprt create_vla(
    const nlohmann::json &element,
    const symbolt *list,
    symbolt *size_var,
    const exprt &list_elem);

  exprt build_list_at_call(
    const exprt &list,
    const exprt &index,
    const nlohmann::json &element);

  symbolt &create_list();

  exprt
  handle_range_slice(const exprt &array, const nlohmann::json &slice_node);

  exprt
  handle_index_access(const exprt &array, const nlohmann::json &slice_node);

  python_converter &converter_;
  const nlohmann::json &list_value_;

  // <list_id, <elem_id, elem_type>>
  static std::unordered_map<std::string, TypeInfo> list_type_map;
};
