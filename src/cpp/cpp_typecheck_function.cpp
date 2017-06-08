/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <util/c_qualifiers.h>
#include <cpp/cpp_template_type.h>
#include <cpp/cpp_type2name.h>
#include <cpp/cpp_typecheck.h>
#include <cpp/cpp_util.h>
#include <util/expr_util.h>
#include <util/i2string.h>

/*******************************************************************\

Function: cpp_typecheckt::convert_argument

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::convert_argument(
  const irep_idt &mode,
  code_typet::argumentt &argument)
{
  std::string identifier=id2string(argument.get_identifier());

  if(identifier.empty())
  {
    identifier="#anon_arg"+i2string(anon_counter++);
    argument.set_base_name(identifier);
  }

  identifier=cpp_scopes.current_scope().prefix+
             id2string(identifier);

  argument.set_identifier(identifier);

  symbolt symbol;

  symbol.name=identifier;
  symbol.base_name=argument.get_base_name();
  symbol.location=argument.location();
  symbol.mode=mode;
  symbol.module=module;
  symbol.type=argument.type();
  symbol.lvalue=!is_reference(symbol.type);

  assert(!symbol.base_name.empty());

  symbolt *new_symbol;

  if(context.move(symbol, new_symbol))
  {
    err_location(symbol.location);
    str << "cpp_typecheckt::convert_argument: context.move("
        << symbol.name << ") failed";
    throw 0;
  }

  // put into scope
  cpp_scopes.put_into_scope(*new_symbol);
}

/*******************************************************************\

Function: cpp_typecheckt::convert_arguments

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::convert_arguments(
  const irep_idt &mode,
  code_typet &function_type)
{
  code_typet::argumentst &arguments=
    function_type.arguments();

  for(code_typet::argumentst::iterator
      it=arguments.begin();
      it!=arguments.end();
      it++)
    convert_argument(mode, *it);
}

/*******************************************************************\

Function: cpp_typecheckt::convert_function

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::convert_function(symbolt &symbol)
{
  code_typet &function_type=
    to_code_type(template_subtype(symbol.type));

  // Is this a template that was instantiated for a function overload, but isn't
  // referred to? If so, don't attempt to convert it, because the template
  // itself would never be instantiated in a real compilation. This is the tail
  // end of SFINAE, but instead of discarding compilation errors in unused
  // templates, we just don't convert them.
  if (symbol.value.get("#speculative_template") == "1" &&
      symbol.value.get("#template_in_use") != "1")
    return;

  // only a prototype?
  if(symbol.value.is_nil())
    return;

  // if it is a destructor, add the implicit code
  if(symbol.type.get("return_type")=="destructor")
  {
    const symbolt &msymb=lookup(symbol.type.get("#member_name"));

    assert(symbol.value.id()=="code");
    assert(symbol.value.statement() == "block");

    // vtables should be updated as soon as the destructor is called
    // dtors contains the destructors for members and base classes,
    // that should be called after the code of the current destructor
    code_blockt vtables, dtors;
    dtor(msymb, vtables, dtors);

    if(vtables.has_operands())
      symbol.value.operands().insert(symbol.value.operands().begin(), vtables);

    if(dtors.has_operands())
      symbol.value.copy_to_operands(dtors);
  }

  // enter appropriate scope
  cpp_save_scopet saved_scope(cpp_scopes);
  cpp_scopet &function_scope=cpp_scopes.set_scope(symbol.name);

  // fix the scope's prefix
  function_scope.prefix+=id2string(symbol.name)+"::";

  // genuine function definition -- do the parameter declarations
  convert_arguments(symbol.mode, function_type);

  // create "this" if it's a non-static method
  if(function_scope.is_method &&
     !function_scope.is_static_member)
  {
    code_typet::argumentst &arguments=function_type.arguments();
    assert(arguments.size()>=1);
    code_typet::argumentt &this_argument_expr=arguments.front();
    function_scope.this_expr=exprt("symbol", this_argument_expr.type());
    function_scope.this_expr.identifier(this_argument_expr.cmt_identifier());
  }
  else
    function_scope.this_expr.make_nil();

  // do the function body
  start_typecheck_code();

  // save current return type
  typet old_return_type=return_type;

  return_type=function_type.return_type();

  // constructor, destructor?
  if(return_type.id()=="constructor" ||
     return_type.id()=="destructor")
    return_type=empty_typet();

  typecheck_code(to_code(symbol.value));

  symbol.value.type()=symbol.type;

  return_type = old_return_type;
}

/*******************************************************************\

Function: cpp_typecheckt::function_identifier

  Inputs:

 Outputs:

 Purpose: for function overloading

\*******************************************************************/

irep_idt cpp_typecheckt::function_identifier(const typet &type)
{
  const code_typet &function_type=
    to_code_type(template_subtype(type));

  const code_typet::argumentst &arguments=
    function_type.arguments();

  std::string result;
  bool first=true;

  result+='(';

  // the name of the function should not depend on
  // the class name that is encoded in the type of this,
  // but we must distinguish "const" and "non-const" member
  // functions

  code_typet::argumentst::const_iterator it=
    arguments.begin();

  if(it!=arguments.end() &&
     it->get_identifier()=="this")
  {
    const typet &pointer=it->type();
    const typet &symbol =pointer.subtype();
    if(symbol.cmt_constant()) result+="const$";
    if(symbol.cmt_volatile()) result+="volatile$";
    result+="this";
    first=false;
    it++;
  }

  // we skipped the "this", on purpose!

  for(; it!=arguments.end(); it++)
  {
    if(first) first=false; else result+=",";
    typet tmp_type=it->type();
    result+=cpp_type2name(it->type());
  }

  result+=')';

  return result;
}
