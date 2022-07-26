#include <util/replace_symbol.h>
#include <util/std_types.h>

bool replace_symbolt::replace(exprt &dest)
{
  if(dest.id() == "symbol")
  {
    expr_mapt::const_iterator it = expr_map.find(dest.identifier());

    if(it != expr_map.end())
    {
      dest = it->second;
      return false;
    }
  }

  bool result = true;

  Forall_operands(it, dest)
    result = replace(*it) && result;

  result = replace(dest.type()) && result;

  return result;
}

bool replace_symbolt::replace(typet &dest)
{
  if(dest.has_subtype())
    replace(dest.subtype());

  Forall_subtypes(it, dest)
    replace(*it);

  if(dest.id() == "struct" || dest.id() == "union")
  {
    struct_typet &struct_type = to_struct_type(dest);
    struct_typet::componentst &components = struct_type.components();
    for(auto &component : components)
      replace(component);
  }
  else if(dest.is_code())
  {
    code_typet &code_type = to_code_type(dest);
    code_typet::argumentst &arguments = code_type.arguments();
    for(auto &argument : arguments)
      replace(argument);
  }

  if(dest.id() == "symbol")
  {
    type_mapt::const_iterator it = type_map.find(dest.identifier());

    if(it != type_map.end())
    {
      dest = it->second;
      return false;
    }
  }

  return true;
}
