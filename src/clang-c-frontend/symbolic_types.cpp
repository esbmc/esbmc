#include "symbolic_types.h"

void get_complete_struct_type(struct_typet &type, const namespacet &ns)
{
  for(auto &comp : type.components())
  {
    typet &it_type = comp.type();
    if(it_type.is_symbol())
    {
      const symbolt *complete_type = ns.lookup(it_type.identifier());
      assert(complete_type);
      if(complete_type->type.is_struct())
      {
        it_type = complete_type->type;
        // follow the type recursively to replace all symbolic types
        get_complete_struct_type(to_struct_type(it_type), ns);
      }
    }
  }
}

bool array_type_contains_symbolic(
  const typet &type,
  typet &complete_type,
  const namespacet &ns)
{
  // check whether we have symbolic types in array type
  if(type.id() == "array")
  {
    complete_type = type;
    typet &sub_type = complete_type.subtype();
    if(sub_type.is_symbol())
    {
      const symbolt *the_type = ns.lookup(sub_type.identifier());
      assert(the_type);
      if(the_type->type.is_struct())
      {
        // replace the symbolic subtype then
        // follow the type recursively to replace all symbolic types
        sub_type = the_type->type;
        get_complete_struct_type(to_struct_type(sub_type), ns);
      }
    }
    return true;
  }

  return false;
}