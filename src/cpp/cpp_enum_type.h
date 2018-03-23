/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_ENUM_TYPE_H
#define CPROVER_CPP_ENUM_TYPE_H

#include <cassert>
#include <util/config.h>
#include <util/type.h>

class cpp_enum_typet : public typet
{
public:
  cpp_enum_typet() : typet("c_enum")
  {
    set("width", config.ansi_c.int_width);
  }

  const irep_idt &get_name() const
  {
    return get("name");
  }

  void set_name(const irep_idt &name)
  {
    set("name", name);
  }

  const irept &body() const
  {
    return find("body");
  }

  irept &body()
  {
    return add("body");
  }

  bool has_body() const
  {
    return find("body").is_not_nil();
  }

  bool get_tag_only_declaration() const
  {
    return get_bool("#tag_only_declaration");
  }
};

extern inline const cpp_enum_typet &to_cpp_enum_type(const irept &irep)
{
  assert(irep.id() == "c_enum");
  return static_cast<const cpp_enum_typet &>(irep);
}

extern inline cpp_enum_typet &to_cpp_enum_type(irept &irep)
{
  assert(irep.id() == "c_enum");
  return static_cast<cpp_enum_typet &>(irep);
}

#endif
