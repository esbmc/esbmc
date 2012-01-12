/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <i2string.h>
#include <expr_util.h>
#include <location.h>

#include "cpp_typecheck.h"
#include "cpp_convert_type.h"
#include "cpp_declarator_converter.h"
#include "cpp_template_type.h"
#include "cpp_util.h"

/*******************************************************************\

Function: cpp_typecheckt::typecheck_code

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::typecheck_code(codet &code)
{
  const irep_idt &statement=code.get("statement");

  if(statement=="throw")
  {
    code.type()=typet("code");
    typecheck_throw(code);
  }
  else if(statement=="member_initializer")
  {
    code.type()=typet("code");
    typecheck_member_initializer(code);
  }
  else
    c_typecheck_baset::typecheck_code(code);
}

/*******************************************************************\

Function: cpp_typecheckt::typecheck_throw

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::typecheck_throw(codet &code)
{
}

/*******************************************************************\

Function: cpp_typecheckt::typecheck_member_initializer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::typecheck_member_initializer(codet &code)
{
  std::string identifier, base_name;
  const cpp_namet &member=static_cast<cpp_namet &>(code.add("member"));

#if 0
  member.convert(identifier, base_name);

  if(identifier!=base_name)
  {
    err_location(code);
    str << "only simple member identifiers allowed here" << std::endl;
    throw 0;
  }

  if(cpp_scopes.current_scope().this_expr.is_nil())
  {
    err_location(code);
    str << "no member initializer expected here" << std::endl;
    throw 0;
  }
#endif

  // The initializer may be a data member
  // or a parent class
  cpp_typecheck_fargst fargs;
  exprt symbol_expr=
    resolve(member, cpp_typecheck_resolvet::BOTH, fargs);

  if(symbol_expr.id() == "type" &&
     follow(symbol_expr.type()).id() == "struct")
  {
    // it's a constructor
    side_effect_expr_function_callt function_call;
    function_call.location() = code.location();
    cpp_namet func_name = member;
    function_call.function().swap(func_name);

    function_call.arguments().reserve(code.operands().size()+1);
    function_call.arguments().insert(function_call.arguments().begin(),code.operands().begin(),code.operands().end());

    // disbale access control
    bool old_access_control = disable_access_control;
    disable_access_control = true;
    typecheck_expr(function_call);
    disable_access_control = old_access_control;

    assert(function_call.get("statement") == "temporary_object");

    if(function_call.get_bool("#not_accessible"))
    {
      irep_idt access = function_call.get("#access");
      assert( access == "private"
             || access == "protected"
             || access == "noaccess" );

      if(access == "private" || access == "noaccess")
      {
        err_location(code.location());
        str << "error: constructor of `" << symbol_expr.type().get("identifier").as_string()
            << "' is not accessible";
        throw 0;
      }
    }

    // replace the temporary with the current object
    side_effect_expr_function_callt& initializer =
      to_side_effect_expr_function_call(
           static_cast<exprt&>(function_call.add("initializer")).op0());

    exprt& tmp_this = initializer.arguments().front();
    exprt this_expr = cpp_scopes.current_scope().this_expr;
    assert(this_expr.is_not_nil());

    make_ptr_typecast(this_expr,tmp_this.type());
    tmp_this.swap(this_expr);
    code = code_expressiont();
    code.move_to_operands(initializer);
  }
  else
  {
    if(symbol_expr.id() == "dereference" &&
       symbol_expr.op0().id() == "member" &&
       symbol_expr.get_bool("#implicit") == true)
    {
      // treat references as normal pointers
      exprt tmp = symbol_expr.op0();
      symbol_expr.swap(tmp);
    }

    if(symbol_expr.id() == "symbol" &&
       symbol_expr.type().id()!="code")
    {
      // maybe the name of the member collides with a parameter of the constructor
      symbol_expr.make_nil();
      cpp_typecheck_fargst fargs;
      exprt dereference("dereference", cpp_scopes.current_scope().this_expr.type().subtype());
      dereference.copy_to_operands(cpp_scopes.current_scope().this_expr);
      fargs.add_object(dereference);

      {
        cpp_save_scopet cpp_saved_scope(cpp_scopes);
        cpp_scopes.go_to(*(cpp_scopes.id_map[cpp_scopes.current_scope().class_identifier]));
        symbol_expr=resolve(member, cpp_typecheck_resolvet::VAR, fargs);
      }

      if(symbol_expr.id() == "dereference" &&
         symbol_expr.op0().id() == "member" &&
         symbol_expr.get_bool("#implicit") == true)
      {
        // treat references as normal pointers
        exprt tmp = symbol_expr.op0();
        symbol_expr.swap(tmp);
      }
    }

    if(symbol_expr.id() == "member" &&
       symbol_expr.op0().id() == "dereference" &&
       symbol_expr.op0().op0() == cpp_scopes.current_scope().this_expr)
    {
      if(is_reference(symbol_expr.type()))
      {
        // it's a reference member
        if(code.operands().size()!=1)
        {
          err_location(code);
          str << " reference `" + base_name + "' expects one initializer";
          throw 0;
        }
        typecheck_expr(code.op0());
        reference_initializer(code.op0(),symbol_expr.type());

        // assign the pointers
        symbol_expr.type().remove("#reference");
        symbol_expr.set("#lvalue",true);
        code.op0().type().remove("#reference");

        side_effect_exprt assign("assign");
        assign.location() = code.location();
        assign.copy_to_operands(symbol_expr, code.op0());
        typecheck_side_effect_assignment(assign);
        code_expressiont new_code;
        new_code.move_to_operands(assign);
        code.swap(new_code);
      }
      else
      {
        // it's a data member
        already_typechecked(symbol_expr);
        exprt call=
          cpp_constructor(code.location(), symbol_expr, code.operands());

        if(call.is_nil())
        {
          call=codet("skip");
          call.location()=code.location();
        }

        code.swap(call);
      }
    }
    else
    {
      err_location(code);
      str << "invalid member initializer `" << base_name << "'";
      throw 0;
    }
  }
}

/*******************************************************************\

Function: cpp_typecheckt::typecheck_decl

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::typecheck_decl(codet &code)
{
  if(code.operands().size()!=1)
  {
    err_location(code);
    throw "declaration expected to have one operand";
  }

  assert(code.op0().id()=="cpp-declaration");

  cpp_declarationt &declaration=
    to_cpp_declaration(code.op0());

  bool is_typedef=
    convert_typedef(declaration.type());

  typecheck_type(declaration.type());
  assert(declaration.type().is_not_nil());

  if(declaration.declarators().empty() &&
     follow(declaration.type()).get_bool("#is_anonymous"))
  {
    if(follow(declaration.type()).id()!="union")
    {
      err_location(code);
      throw "declaration statement does not declare anything";
    }

    convert_anonymous_union(declaration, code);
    return;
  }

  codet new_code("decl-block");
  new_code.reserve_operands(declaration.declarators().size());

  // Do the declarators (optional).
  Forall_cpp_declarators(it, declaration)
  {
    cpp_declaratort &declarator=*it;
    cpp_declarator_convertert cpp_declarator_converter(*this);

    cpp_declarator_converter.is_typedef=is_typedef;

    const symbolt &symbol=
      cpp_declarator_converter.convert(declaration, declarator);

    if(is_typedef) continue;

    codet decl_statement("decl");
    decl_statement.reserve_operands(2);
    decl_statement.location()=symbol.location;
    decl_statement.copy_to_operands(cpp_symbol_expr(symbol));

    // Do we have an initializer that's not code?
    if(symbol.value.is_not_nil() &&
       symbol.value.id()!="code")
    {
      decl_statement.copy_to_operands(symbol.value);
      assert(follow(decl_statement.op1().type())==follow(symbol.type));
    }

    new_code.move_to_operands(decl_statement);

    // is there a constructor to be called?
    if(symbol.value.is_not_nil())
    {
      assert(it->find("init_args").is_nil());
      if(symbol.value.id()=="code")
        new_code.copy_to_operands(symbol.value);
    }
    else
    {
      exprt object_expr=cpp_symbol_expr(symbol);

      already_typechecked(object_expr);

      exprt constructor_call=
        cpp_constructor(
          symbol.location,
          object_expr,
          declarator.init_args().operands());

      if(constructor_call.is_not_nil())
        new_code.move_to_operands(constructor_call);
    }
  }

  code.swap(new_code);
}

/*******************************************************************\

Function: cpp_typecheck_codet::typecheck_block

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::typecheck_block(codet &code)
{
  cpp_save_scopet saved_scope(cpp_scopes);
  cpp_scopes.new_block_scope();

  c_typecheck_baset::typecheck_block(code);
}

/*******************************************************************\

Function: cpp_typecheckt::typecheck_assign

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::typecheck_assign(codet &code)
{

  if(code.operands().size()!=2)
    throw "assignment statement expected to have two operands";

  // turn into a sideeffect
  side_effect_exprt expr(code.get("statement"));
  expr.operands() = code.operands();
  typecheck_expr(expr);

  code_expressiont code_expr;
  code_expr.copy_to_operands(expr);
  code_expr.location() = code.location();

  code.swap(code_expr);
}
