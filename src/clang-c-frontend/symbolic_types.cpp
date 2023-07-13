#include "symbolic_types.h"

void get_complete_struct_type(struct_typet &type, const namespacet &ns)
{
  for(auto &comp : type.components())
  {
    typet &it_type = comp.type();
    if(it_type.is_symbol())
    {
      const symbolt *actual_type = ns.lookup(it_type.identifier());
      assert(actual_type);
      if(actual_type->type.is_struct())
      {
        it_type = actual_type->type;
        // follow the type recursively to replace all symbolic types
        get_complete_struct_type(to_struct_type(it_type), ns);
      }
    }
  }
}