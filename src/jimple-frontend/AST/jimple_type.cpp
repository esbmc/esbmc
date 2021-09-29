//
// Created by rafaelsamenezes on 21/09/2021.
//

#include <jimple-frontend/AST/jimple_type.h>

void jimple_type::from_json(const json &j)
{
  // Non-void type
  // TODO: j.at_to("mode")
  j.at("identifier").get_to(name);
  j.at("dimensions").get_to(dimensions);
  j.at("mode").get_to(mode);

}

typet jimple_type::get_base_type() const {
  // TODO: Hash-Table
  // TODO: Type table
  if(mode != "basic")
    return struct_union_typet(name);

  // This is the equivalent to C
  if(name == "void")
  {
    return empty_typet();
  }
  else if(name == "int")
  {
    return int_type();
  }
  else {
    return typet(name);
  }
}

typet jimple_type::to_typet() const {
  if(is_array())
    return get_arr_type();
  return get_base_type();
}

std::string jimple_type::to_string() const
{
  std::ostringstream oss;
  oss << "Type: " << name
      << " [" << dimensions << "]";
  return oss.str();
}
