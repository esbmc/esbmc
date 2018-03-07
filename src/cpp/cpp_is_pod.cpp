/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cpp/cpp_typecheck.h>

bool cpp_typecheckt::cpp_is_pod(const typet &type) const
{
  if(type.id() == "struct")
  {
    // Not allowed in PODs:
    // * Non-PODs
    // * Constructors/Destructors
    // * virtuals
    // * private/protected, unless static
    // * overloading assignment operator
    // * Base classes

    // XXX jmorse: certain things listed above don't always make their way into
    // the class definition though, such as templated constructors. In that
    // case, we set a flag to indicate that such methods have been seen, before
    // removing them. The "is_not_pod" flag thus only guarentees that it /isn't/
    // and its absence doesn't guarentee that it is.
    if(!type.find("is_not_pod").is_nil())
      return false;

    const struct_typet &struct_type = to_struct_type(type);

    if(!type.find("bases").get_sub().empty())
      return false;

    const struct_typet::componentst &components = struct_type.components();

    for(const auto &component : components)
    {
      if(component.is_type())
        continue;

      if(component.get_base_name() == "operator=")
        return false;

      if(component.get_bool("is_virtual"))
        return false;

      const typet &sub_type = component.type();

      if(sub_type.id() == "code")
      {
        if(component.get_bool("is_virtual"))
          return false;

        const typet &return_type = to_code_type(sub_type).return_type();

        if(
          return_type.id() == "constructor" || return_type.id() == "destructor")
          return false;
      }
      else if(
        component.get("access") != "public" && !component.get_bool("is_static"))
        return false;

      if(!cpp_is_pod(sub_type))
        return false;
    }

    return true;
  }
  if(type.id() == "array")
  {
    return cpp_is_pod(type.subtype());
  }
  else if(type.id() == "pointer")
  {
    if(is_reference(type)) // references are not PODs
      return false;

    // but pointers are PODs!
    return true;
  }
  else if(type.id() == "symbol")
  {
    const symbolt &symb = lookup(type.identifier());
    assert(symb.is_type);
    return cpp_is_pod(symb.type);
  }

  // everything else is POD
  return true;
}
