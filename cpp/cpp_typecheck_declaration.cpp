/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\********************************************************************/

#include "cpp_typecheck.h"
#include "cpp_declarator_converter.h"
#include <i2string.h>
#include <expr_util.h>

/*******************************************************************\

Function: cpp_typecheckt::convert_typedef

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cpp_typecheckt::convert_typedef(typet &type)
{
  if(type.id()=="merged_type" &&
     type.subtypes().size()>=2 &&
     type.subtypes()[0].id()=="typedef")
  {
    type.subtypes().erase(type.subtypes().begin());
    return true;
  }

  return false;
}

/*******************************************************************\

Function: cpp_typecheckt::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::convert(cpp_declarationt &declaration)
{
  function_bodiest old_function_bodies = function_bodies;
  function_bodies.clear();

  // see if it's empty
  if(declaration.find("type").is_nil() &&
     !declaration.has_operands())
    return;

  if(declaration.get_bool("is_template"))
  {
    convert_template_declaration(declaration);
  }
  else
  {
    // do the first part, the type

    typet &type=declaration.type();
    bool is_typedef=convert_typedef(type);

    typecheck_type(type);

    if(declaration.declarators().empty()
     && follow(declaration.type()).get_bool("#is_anonymous"))
    {
      typet final_type = follow(declaration.type());
      if(final_type.id() != "union")
      {
        err_location(final_type.location());
        throw "declaration does not declare anything";
      }
      codet dummy;
      convert_anonymous_union(declaration,dummy);
    }

    // do the declarators (optional)
    Forall_cpp_declarators(it, declaration)
    {
      cpp_declaratort declarator = *it;

      cpp_declarator_convertert cpp_declarator_converter(*this);

      cpp_declarator_converter.is_typedef=is_typedef;

      symbolt& symb = cpp_declarator_converter.convert(
        type, declaration.storage_spec(),
        declaration.member_spec(), declarator);
      exprt symb_expr = cpp_symbol_expr(symb);
      it->swap(symb_expr);


      // is there a constructor to be called?
      if(symb.lvalue && declarator.init_args().has_operands())
      {
          symb.value =
            cpp_constructor(
              symb.location,
              cpp_symbol_expr(symb),
              declarator.init_args().operands());
       }
      }
  }

  typecheck_function_bodies();
  function_bodies = old_function_bodies;

}

/*******************************************************************\

Function: cpp_typecheckt::convert_anonymous_union

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::convert_anonymous_union(
   cpp_declarationt &declaration,
   codet& code)
{
    codet new_code("decl-block");
    new_code.reserve_operands(declaration.declarators().size());

    // unnamed object
    std::string identifier = "#anon"+i2string(anon_counter++);
    irept name("name");
    name.set("identifier",identifier);
    cpp_namet cpp_name;
    cpp_name.move_to_sub(name);
    cpp_declaratort declarator;
    declarator.name() = cpp_name;

    cpp_declarator_convertert cpp_declarator_converter(*this);

    const symbolt &symbol=
      cpp_declarator_converter.convert(declaration, declarator);

    if(!cpp_is_pod(declaration.type()))
    {
     err_location(follow(declaration.type()).location());
     str << "anonymous union is not POD";
     throw 0;
    }

    codet decl_statement("decl");
    decl_statement.reserve_operands(2);
    decl_statement.copy_to_operands(cpp_symbol_expr(symbol));

    new_code.move_to_operands(decl_statement);

    // do scoping
    symbolt union_symbol = context.symbols[follow(symbol.type).get("name")];
    const irept::subt &components = union_symbol.type.add("components").get_sub();
    forall_irep(it, components)
    {
      if(it->find("type").id()=="code")
      {
       err_location(union_symbol.type.location());
       str << "anonymous union " << union_symbol.base_name
           << " shall not have function members\n";
       throw 0;
      }

      const irep_idt& base_name = it->get("base_name");

      if(cpp_scopes.current_scope().contains(base_name))
      {
        str << "`" << base_name << "' already in scope";
        throw 0;
      }

     cpp_idt &id=cpp_scopes.current_scope().insert(base_name);
     id.id_class = cpp_idt::SYMBOL;
     id.identifier=it->get("name");
     id.class_identifier=union_symbol.name;
     id.is_member=true;
   }
    context.symbols[union_symbol.name].type.set("#unnamed_object", symbol.base_name);

    code.swap(new_code);
}
