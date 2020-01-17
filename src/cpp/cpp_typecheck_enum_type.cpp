/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/c_qualifiers.h>
#include <cpp/cpp_enum_type.h>
#include <cpp/cpp_typecheck.h>
#include <util/arith_tools.h>
#include <util/i2string.h>

void cpp_typecheckt::typecheck_enum_body(symbolt &enum_symbol)
{
  typet &type = enum_symbol.type;

  exprt &body = static_cast<exprt &>(type.add("body"));
  irept::subt &components = body.get_sub();

  typet enum_type("symbol");
  enum_type.identifier(enum_symbol.id);

  BigInt i = 0;

  Forall_irep(it, components)
  {
    const irep_idt &name = it->name();

    if(it->find("value").is_not_nil())
    {
      exprt &value = static_cast<exprt &>(it->add("value"));
      typecheck_expr(value);
      make_constant_index(value);
      if(to_integer(value, i))
        throw "failed to produce integer for enum";
    }

    exprt final_value("constant", enum_type);
    final_value.value(integer2string(i));

    symbolt symbol;

    symbol.id = id2string(enum_symbol.id) + "::" + id2string(name);
    symbol.name = name;
    symbol.value.swap(final_value);
    symbol.location = static_cast<const locationt &>(it->find("#location"));
    symbol.mode = "C++"; // All types are c++ types
    symbol.module = module;
    symbol.type = enum_type;
    symbol.is_type = false;
    symbol.is_macro = true;

    symbolt *new_symbol;
    if(context.move(symbol, new_symbol))
      throw "cpp_typecheckt::typecheck_enum_body: context.move() failed";

    cpp_idt &scope_identifier = cpp_scopes.put_into_scope(*new_symbol);

    scope_identifier.id_class = cpp_idt::SYMBOL;

    ++i;
  }
}

void cpp_typecheckt::typecheck_enum_type(typet &type)
{
  // first save qualifiers
  c_qualifierst qualifiers;
  qualifiers.read(type);

  // these behave like special struct types
  // replace by type symbol

  cpp_enum_typet &enum_type = to_cpp_enum_type(type);

  bool has_body = enum_type.has_body();
  std::string base_name = id2string(enum_type.get_name());
  bool anonymous = base_name.empty();
  bool tag_only_declaration = enum_type.get_tag_only_declaration();

  if(anonymous)
    base_name = "#anon" + i2string(anon_counter++);

  cpp_scopet &dest_scope = tag_scope(base_name, has_body, tag_only_declaration);

  const irep_idt symbol_name = dest_scope.prefix + "tag." + base_name;

  // check if we have it

  symbolt *previous_symbol = context.find_symbol(symbol_name);

  if(previous_symbol != nullptr)
  {
    // we do!

    symbolt &symbol = *previous_symbol;

    if(has_body)
    {
      err_location(type);
      str << "error: enum symbol `" << base_name << "' declared previously"
          << std::endl;
      str << "location of previous definition: " << symbol.location
          << std::endl;
      throw 0;
    }
  }
  else if(has_body)
  {
    std::string pretty_name = cpp_scopes.current_scope().prefix + base_name;

    symbolt symbol;

    symbol.id = symbol_name;
    symbol.name = base_name;
    symbol.value.make_nil();
    symbol.location = type.location();
    symbol.mode = "C++"; // All types are c++ types.
    symbol.module = module;
    symbol.type.swap(type);
    symbol.is_type = true;
    symbol.is_macro = false;

    // move early, must be visible before doing body
    symbolt *new_symbol;
    if(context.move(symbol, new_symbol))
      throw "cpp_typecheckt::typecheck_enum_type: context.move() failed";

    // put into scope
    cpp_idt &scope_identifier = cpp_scopes.put_into_scope(*new_symbol);

    scope_identifier.id_class = cpp_idt::CLASS;

    typecheck_enum_body(*new_symbol);
  }
  else
  {
    err_location(type);
    str << "use of enum `" << base_name << "' without previous declaration";
    throw 0;
  }

  // create type symbol
  type = typet("symbol");
  type.identifier(symbol_name);
  qualifiers.write(type);
}
