//
// Created by rafaelsamenezes on 21/09/2021.
//

#include <jimple-frontend/AST/jimple_type.h>

void jimple_type::from_json(const json &j)
{
  j.at("identifier").get_to(name);
  j.at("dimensions").get_to(dimensions);

  bt = from_map.count(name) != 0 ? from_map[name] : BASE_TYPES::OTHER;
}

typet jimple_type::get_base_type() const
{
  switch(bt)
  {
    case BASE_TYPES::INT:
      return int_type();

    case BASE_TYPES::VOID:
      return empty_typet();

    case BASE_TYPES::OTHER:
      return struct_union_typet(name);

    default:
      return typet(name);
  }
}

typet jimple_type::to_typet() const
{
  if(is_array())
    return get_arr_type();
  return get_base_type();
}

std::string jimple_type::to_string() const
{
  std::ostringstream oss;
  oss << "Type: " << name << " [" << dimensions << "]";
  return oss.str();
}
