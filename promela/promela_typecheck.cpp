/*******************************************************************\

Module: SpecC Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <expr_util.h>
#include <arith_tools.h>
#include <i2string.h>

#include <ansi-c/c_typecast.h>

#include "promela_typecheck.h"
#include "expr2promela.h"
//#include "promela_convert_type.h"

/*******************************************************************\

Function: promela_typecheckt::to_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string promela_typecheckt::to_string(const exprt &expr)
{
  std::string result;
  //expr2promela(expr, result);
  return result;
}

/*******************************************************************\

Function: promela_typecheckt::to_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string promela_typecheckt::to_string(const typet &type)
{
  std::string result;
  //type2promela(type, result);
  return result;
}

/*******************************************************************\

Function: promela_typecheckt::convert_parameters

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

#if 0
void promela_typecheckt::convert_parameters(const exprt &declaration,
                                          symbolt &symbol)
{  
  irept &argument_types=symbol.type.add("argument_types");
  Forall_irep(it, argument_types.get_sub())
  {
    irept &p_declaration=it->find("#declaration");
    if(p_declaration.is_nil())
    {
      err_location(symbol.location);
      err << "no identifier found in parameter declaration" << std::endl;
      throw 0;
    }

    // fix the scope of the argument declaration
    p_declaration.set("scope",
      declaration.get("scope")+declaration.get("name")+"::");

    const symbolt &p_symbol=convert_declaration((exprt &)p_declaration);
    it->remove("#declaration");
    it->set("#identifier", p_symbol.name);
  }
}
#endif

/*******************************************************************\

Function: promela_typecheckt::convert_declaration

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

#if 0
const symbolt &promela_typecheckt::convert_declaration(exprt &declaration)
{
  symbolt symbol;

  symbol.name=std::string("promela::")+
              declaration.get("scope")+declaration.get("name");

  symbol.base_name=declaration.get("name");
  symbol.value.swap(declaration.add("value"));
  symbol.location.swap(declaration.add("#location"));
  symbol.mode="SpecC";
  symbol.module=module;
  symbol.type.swap(declaration.type());

  // do the type
  try
  {
    promela_convert_type(symbol.type);
    typecheck_type(symbol.type);
  }

  catch(const std::string &error)
  {
    err_location(declaration.type());
    err << error << std::endl;
    throw 0;
  }

  switch(atoi(declaration.get("id_class").c_str()))
  {
   case SC_TYPEDEF:
    symbol.is_type=true;
    symbol.is_macro=true;
    break;

   default:
    break;
  }

  if(!symbol.is_type)
  {
    if(symbol.type.id()!="code" &&
       symbol.type.id()!="promela_event")
    {
      // it is a variable
      symbol.is_statevar=true;
      symbol.lvalue=true;

      if(symbol.type.get_bool("#static") ||
         declaration.get("scope").empty()) // global?
        symbol.is_static=true;
    }

    if(symbol.type.id()=="code")
      symbol.theorem=true;
  }

  // move early, it must be visible before doing any value

  symbolt *new_symbol;

  if(context.move(symbol, new_symbol))
    throw "context.move() failed";

  // do the value

  if(!new_symbol->is_type)
  {
    if(new_symbol->type.id()!="code" &&
       new_symbol->type.id()!="promela_event")
    {
      // intitializer
      if(new_symbol->value.is_nil())
      {
        if(new_symbol->is_static)
        {
          new_symbol->value.type()=new_symbol->type;

          if(zero_initializer(new_symbol->value))
          {
            err << "cannot zero-initialize variable of type "
                << to_string(new_symbol->type)
                << std::endl;
            throw 0;
          }            
        }
      }
      else
      {
        typecheck_expr(new_symbol->value);
        if(c_implicit_typecast(new_symbol->value, new_symbol->type))
        {
          err_location(new_symbol->value);
          err << "cannot initialize variable of type "
              << to_string(new_symbol->type)
              << " with value of type "
              << to_string(new_symbol->value.type())
              << std::endl;
          throw 0;
        }
      }
    }

    if(new_symbol->type.id()=="code" &&
       !new_symbol->value.is_nil())
    {
      local_identifiers.clear();

      // true function definition -- do the parameter declarations
      convert_parameters(declaration, *new_symbol);
      typecheck_code(new_symbol->value);
      new_symbol->value.type()=new_symbol->type;

      // it's a function, do local_identifiers
      irept &dest_local_identifiers=new_symbol->value.add("local_identifiers");
      
      for(std::set<std::string>::const_iterator it=local_identifiers.begin();
          it!=local_identifiers.end(); it++)
        dest_local_identifiers.set(*it, "");
    }
  }

  return *new_symbol;
}
#endif

/*******************************************************************\

Function: promela_typecheckt::zero_initializer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

#if 0
bool promela_typecheckt::zero_initializer(exprt &value, const typet &type) const
{
  if(type.id()=="promela_behavior" ||
     type.id()=="promela_channel" ||
     type.id()=="promela_interface")
  {
    const irept::subt &components=
      type.find("components").get_sub();

    value.id("promela_behavior");

    forall_irep(it, components)
    {
      exprt tmp;

      tmp.type()=(typet &)it->find("type");
      if(zero_initializer(tmp))
        return true;

      value.move_to_operands(tmp);
    }

    return false;
  }

  return c_typecheck_baset::zero_initializer(value, type);
}
#endif

/*******************************************************************\

Function: promela_typecheckt::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

#if 0
void promela_typecheckt::typecheck()
{
  for(promela_parset::declarationst::iterator
      it=promela_parse.declarations.begin();
      it!=promela_parse.declarations.end();
      it++)
    convert_declaration(*it);
}
#endif

/*******************************************************************\

Function: promela_typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool promela_typecheck(promela_parse_treet &promela_parse_tree,
                       contextt &context,
                       const std::string &module,
                       std::ostream &err)
{
  promela_typecheckt promela_typecheck(promela_parse_tree, context, module, err);
  return promela_typecheck.typecheck_main();
}

/*******************************************************************\

Function: promela_typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool promela_typecheck(exprt &expr,
                       std::ostream &err,
                       const namespacet &ns)
{
  contextt context;
  promela_parse_treet promela_parse_tree;
  bool result=false;

  promela_typecheckt promela_typecheck(promela_parse_tree, context,
                                       ns.get_context(), "", err);

  try
  {
    promela_typecheck.typecheck_expr(expr);
  }

  catch(int e)
  {
    result=true;
  }

  catch(const char *e)
  {
    err << e << std::endl;
    result=true;
  }

  catch(const std::string &e)
  {
    err << e << std::endl;
    result=true;
  }

  return result;
}
