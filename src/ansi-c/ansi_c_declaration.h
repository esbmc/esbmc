/*******************************************************************\

Module: SpecC Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ANSI_C_DECLARATION_H
#define CPROVER_ANSI_C_DECLARATION_H

#include <cassert>
#include <util/symbol.h>

class ansi_c_declarationt:public exprt
{
public:
  ansi_c_declarationt():exprt("declaration")
  {
  }

  exprt &decl_value()
  {
    return static_cast<exprt &>(add("decl_value"));
  }

  const exprt &decl_value() const
  {
    return static_cast<const exprt &>(irept::decl_value());
  }

  void set_name(const irep_idt &name)
  {
    return this->name(name);
  }

  irep_idt get_name() const
  {
    return get("name");
  }

  irep_idt get_base_name() const
  {
    return this->base_name();
  }

  void set_base_name(const irep_idt &base_name)
  {
    return this->base_name(base_name);
  }

  bool get_is_type() const
  {
    return is_type();
  }

  void set_is_type(bool is_type)
  {
    this->is_type(is_type);
  }

  bool get_is_typedef() const
  {
    return get_bool("is_typedef");
  }

  void set_is_typedef(bool is_typedef)
  {
    set("is_typedef", is_typedef);
  }

  bool get_is_macro() const
  {
    return this->is_macro();
  }

  void set_is_macro(bool is_macro)
  {
    this->is_macro(is_macro);
  }

  bool get_is_static() const
  {
    return get_bool("is_static");
  }

  void set_is_static(bool is_static)
  {
    set("is_static", is_static);
  }

  bool get_is_argument() const
  {
    return get_bool("is_argument");
  }

  void set_is_argument(bool is_argument)
  {
    set("is_argument", is_argument);
  }

  bool get_is_global() const
  {
    return get_bool("is_global");
  }

  void set_is_global(bool is_global)
  {
    set("is_global", is_global);
  }

  bool get_is_register() const
  {
    return get_bool("is_register");
  }

  void set_is_register(bool is_register)
  {
    set("is_register", is_register);
  }

  bool get_is_inline() const
  {
    return get_bool("is_inline");
  }

  void set_is_inline(bool is_inline)
  {
    set("is_inline", is_inline);
  }

  bool get_is_extern() const
  {
    return is_extern();
  }

  void set_is_extern(bool is_extern)
  {
    this->is_extern(is_extern);
  }

  void to_symbol(symbolt &symbol) const
  {
    symbol.clear();
    symbol.location=location();
    symbol.value=decl_value();
    symbol.type=type();
    symbol.name=get_name();
    symbol.base_name=get_base_name();
    symbol.is_type=get_is_type();
    symbol.is_extern=get_is_extern();
    symbol.is_macro=get_is_macro();
    symbol.is_parameter=get_is_argument();

    bool is_code=symbol.type.is_code();

    symbol.static_lifetime=
      !symbol.is_macro &&
      !symbol.is_type &&
      (get_is_global() || get_is_static()) &&
      !is_code;

    symbol.file_local=
      get_is_static() || symbol.is_macro ||
      (!get_is_global() && !get_is_extern() && !is_code);

    if(get_is_inline())
      symbol.type.inlined(true);
  }
};

extern inline ansi_c_declarationt &to_ansi_c_declaration(exprt &expr)
{
  assert(expr.id()=="declaration");
  return static_cast<ansi_c_declarationt &>(expr);
}

extern inline const ansi_c_declarationt &to_ansi_c_declaration(const exprt &expr)
{
  assert(expr.id()=="declaration");
  return static_cast<const ansi_c_declarationt &>(expr);
}

#endif
