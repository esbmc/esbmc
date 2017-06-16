/*******************************************************************\

Module: SPEC-C Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/ansi_c_convert.h>
#include <ansi-c/ansi_c_convert_type.h>
#include <ansi-c/ansi_c_declaration.h>
#include <util/config.h>
#include <util/std_types.h>

/*******************************************************************\

Function: ansi_c_convertt::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_convertt::convert(ansi_c_parse_treet &ansi_c_parse_tree)
{
  for(ansi_c_parse_treet::declarationst::iterator
      it=ansi_c_parse_tree.declarations.begin();
      it!=ansi_c_parse_tree.declarations.end();
      ++it)
    convert_declaration(*it);
}

/*******************************************************************\

Function: ansi_c_convertt::convert_declaration

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_convertt::convert_declaration(ansi_c_declarationt &declaration)
{
  c_storage_spect c_storage_spec;

  convert_type(declaration.type(), c_storage_spec);

  declaration.set_is_inline(c_storage_spec.is_inline);
  declaration.set_is_static(c_storage_spec.is_static);
  declaration.set_is_extern(c_storage_spec.is_extern);
  declaration.set_is_register(c_storage_spec.is_register);

  // do not overwrite is_typedef -- it's done by the parser
  // typedefs are macros
  if(declaration.get_is_typedef())
    declaration.set_is_macro(true);

  // add language prefix
  declaration.set_name(id2string(declaration.get_name()));

  if(declaration.decl_value().is_not_nil())
  {
    if(declaration.type().is_code())
      convert_code(to_code(declaration.decl_value()));
    else
      convert_expr(declaration.decl_value());
  }
}

/*******************************************************************\

Function: ansi_c_convertt::convert_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_convertt::convert_expr(exprt &expr)
{
  Forall_operands(it, expr)
    convert_expr(*it);

  if(expr.id()=="symbol")
  {
    expr.identifier(final_id(expr.identifier()));
    expr.remove("#id_class");
    expr.remove("#base_name");
  }
  else if(expr.id()=="sizeof")
  {
    if(expr.operands().size()==0)
    {
      typet type=static_cast<const typet &>(expr.c_sizeof_type());
      convert_type(type);
      expr.c_sizeof_type(type);
    }
  }
  else if(expr.id()=="builtin_va_arg")
  {
    convert_type(expr.type());
  }
  else if(expr.id()=="builtin_offsetof")
  {
    typet offsetof_type=static_cast<const typet &>(expr.offsetof_type());
    convert_type(offsetof_type);
    expr.offsetof_type(offsetof_type);
  }
  else if(expr.id()=="typecast")
  {
    convert_type(expr.type());
  }
}

/*******************************************************************\

Function: ansi_c_convertt::convert_code

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_convertt::convert_code(codet &code)
{
  const irep_idt &statement=code.get_statement();

  if(statement=="expression")
  {
    assert(code.operands().size()==1);
    convert_expr(code.op0());
  }
  else if(statement=="decl")
  {
    assert(code.operands().size()==1 ||
           code.operands().size()==2);

    convert_expr(code.op0());

    if(code.operands().size()==2)
      convert_expr(code.op1());
  }
  else if(statement=="label")
  {
    assert(code.operands().size()==1);
    convert_code(to_code(code.op0()));
  }
  else if(statement=="switch_case")
  {
    assert(code.operands().size()==2);
    if(code.op0().is_not_nil())
      convert_expr(code.op0());

    convert_code(to_code(code.op1()));
  }
  else if(statement=="block")
  {
    Forall_operands(it, code)
      convert_code(to_code(*it));
  }
  else if(statement=="ifthenelse")
  {
    assert(code.operands().size()==2 ||
           code.operands().size()==3);

    convert_expr(code.op0());
    convert_code(to_code(code.op1()));

    if(code.operands().size()==3)
      convert_code(to_code(code.op2()));
  }
  else if(statement=="while" ||
          statement=="dowhile")
  {
    assert(code.operands().size()==2);

    convert_expr(code.op0());
    convert_code(to_code(code.op1()));
  }
  else if(statement=="for")
  {
    assert(code.operands().size()==4);

    if(code.op0().is_not_nil())
      convert_code(to_code(code.op0()));

    if(code.op1().is_not_nil())
      convert_expr(code.op1());

    if(code.op2().is_not_nil())
    {
      convert_expr(code.op2());
      codet tmp("expression");
      tmp.move_to_operands(code.op2());
      code.op2().swap(tmp);
    }

    convert_code(to_code(code.op3()));
  }
  else if(statement=="msc_try_except")
  {
    assert(code.operands().size()==3);
    convert_code(to_code(code.op0()));
    convert_expr(code.op1());
    convert_code(to_code(code.op2()));
  }
  else if(statement=="msc_try_finally")
  {
    assert(code.operands().size()==2);
    convert_code(to_code(code.op0()));
    convert_code(to_code(code.op1()));
  }
  else if(statement=="switch")
  {
    assert(code.operands().size()==2);

    convert_expr(code.op0());
    convert_code(to_code(code.op1()));
  }
  else if(statement=="break")
  {
  }
  else if(statement=="goto")
  {
  }
  else if(statement=="continue")
  {
  }
  else if(statement=="return")
  {
    if(code.operands().size()==1)
      convert_expr(code.op0());
  }
  else if(statement=="skip")
  {
  }
  else if(statement=="asm")
  {
  }
  else
  {
    err_location(code);
    str << "unexpected statement during conversion: " << statement;
    throw 0;
  }
}

/*******************************************************************\

Function: ansi_c_convertt::convert_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_convertt::convert_type(typet &type)
{
  c_storage_spect c_storage_spec;
  convert_type(type, c_storage_spec);
}

/*******************************************************************\

Function: ansi_c_convertt::convert_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_convertt::convert_type(
  typet &type,
  c_storage_spect &c_storage_spec)
{
  ansi_c_convert_typet ansi_c_convert_type(get_message_handler());

  ansi_c_convert_type.read(type);
  ansi_c_convert_type.write(type);

  c_storage_spec=ansi_c_convert_type.c_storage_spec;

  if(type.id()=="pointer")
  {
    c_storage_spect sub_storage_spec;

    convert_type(type.subtype(), sub_storage_spec);
    c_storage_spec|=sub_storage_spec;
  }
  else if(type.id()=="c_bitfield")
  {
    convert_type(type.subtype());
    exprt tmp = static_cast<const exprt &>(type.size_irep());
    convert_expr(tmp);
    type.size(tmp);
    // XXX jmorse - does this reveal a condition where c_bitfield doesn't have
    // a size field?
  }
  else if(type.id()=="symbol")
  {
    irep_idt identifier=final_id(type.identifier());
    type.identifier(identifier);
    type.remove("#id_class");
    type.remove("#base_name");
  }
  else if(type.id()=="ansi_c_event")
  {
    if(!ansi_c_convert_type.c_qualifiers.is_empty())
    {
      err_location(type);
      throw "no qualifiers permitted for event type";
    }
  }
  else if(type.is_code())
  {
    c_storage_spect sub_storage_spec;

    convert_type(type.subtype(), sub_storage_spec);
    c_storage_spec|=sub_storage_spec;

    code_typet &code_type=to_code_type(type);

    // change subtype to return_type
    code_type.return_type().swap(type.subtype());
    type.remove("subtype");

    // take care of argument types
    code_typet::argumentst &arguments=code_type.arguments();

    // see if we have an ellipsis
    if(!arguments.empty() &&
       arguments.back().id()=="ansi_c_ellipsis")
    {
      code_type.make_ellipsis();
      arguments.pop_back();
    }

    for(code_typet::argumentst::iterator
        it=arguments.begin();
        it!=arguments.end();
        ++it)
    {
      if(it->id()=="declaration")
      {
        code_typet::argumentt argument;

        ansi_c_declarationt &declaration=
          to_ansi_c_declaration(*it);

        convert_type(declaration.type());

        irep_idt base_name=declaration.get_base_name();

        argument.type().swap(declaration.type());
        argument.set_base_name(base_name);
        argument.location()=declaration.location();

        argument.set_identifier(id2string(declaration.get_name()));

        it->swap(argument);
      }
      else if(it->id()=="ansi_c_ellipsis")
        throw "ellipsis only allowed as last argument";
      else
        throw "unexpected argument: "+it->id_string();
    }
  }
  else if(type.is_array())
  {
    array_typet &array_type=to_array_type(type);

    c_storage_spect sub_storage_spec;

    convert_type(array_type.subtype(), sub_storage_spec);
    c_storage_spec|=sub_storage_spec;

    convert_expr(array_type.size());
  }
  else if(type.id()=="incomplete_array")
  {
    c_storage_spect sub_storage_spec;

    convert_type(type.subtype(), sub_storage_spec);
    c_storage_spec|=sub_storage_spec;
  }
  else if(type.id()=="struct" ||
          type.id()=="union" ||
          type.id()=="ansi_c_interface" ||
          type.id()=="ansi_c_channel" ||
          type.id()=="ansi_c_behavior")
  {
    // Create new components subt to operate upon
    irept::subt components=type.components().get_sub();

    Forall_irep(it, components)
    {
      // the arguments are now declarations
      ansi_c_declarationt &component=
        to_ansi_c_declaration((exprt &)*it);

      exprt new_component("component");

      new_component.location()=component.location();
      new_component.name(component.get_base_name());
      new_component.pretty_name(component.get_base_name());
      new_component.type().swap(component.type());

      convert_type(new_component.type());

      component.swap(new_component);
    }

    // Set into type
    irept tmp = type.components();
    tmp.get_sub() = components;
    type.components(tmp);
  }
  else if(type.id()=="type_of")
  {
    if(type.is_expression())
      convert_expr((exprt&)type.subtype());
    else
      convert_type(type.subtype());
  }
  else if(type.id()=="c_enum" ||
          type.id()=="incomplete_c_enum")
  {
    // add width
    type.width(config.ansi_c.int_width);
  }
  else if(type.id()=="void")
  {
    type.id("empty");
  }
}

/*******************************************************************\

Function: ansi_c_convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ansi_c_convert(
  ansi_c_parse_treet &ansi_c_parse_tree,
  const std::string &module,
  message_handlert &message_handler)
{
  ansi_c_convertt ansi_c_convert(module, message_handler);

  try
  {
    ansi_c_convert.convert(ansi_c_parse_tree);
  }

  catch(int e)
  {
    ansi_c_convert.error();
  }

  catch(const char *e)
  {
    ansi_c_convert.error(e);
  }

  catch(const std::string &e)
  {
    ansi_c_convert.error(e);
  }

  return ansi_c_convert.get_error_found();
}

/*******************************************************************\

Function: ansi_c_convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ansi_c_convert(
  exprt &expr,
  const std::string &module,
  message_handlert &message_handler)
{
  ansi_c_convertt ansi_c_convert(module, message_handler);

  try
  {
    ansi_c_convert.convert_expr(expr);
  }

  catch(int e)
  {
    ansi_c_convert.error();
  }

  catch(const char *e)
  {
    ansi_c_convert.error(e);
  }

  catch(const std::string &e)
  {
    ansi_c_convert.error(e);
  }

  return ansi_c_convert.get_error_found();
}
