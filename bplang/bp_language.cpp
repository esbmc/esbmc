/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <expr_util.h>
#include <std_code.h>

#include "bp_language.h"
#include "bp_typecheck.h"
#include "bp_parser.h"
#include "expr2bp.h"

/*******************************************************************\

Function: bp_languaget::final

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bp_languaget::final(
  contextt &context,
  message_handlert &message_handler)
{
  // do main symbol
  
  code_typet main_type;
  main_type.return_type()=empty_typet();
  
  symbolt new_symbol;
  new_symbol.name="main";
  new_symbol.type=main_type;
  
  const symbolst::const_iterator s_it=
    context.symbols.find("bp::fkt::main");
    
  if(s_it==context.symbols.end())
    return false;
    
  const symbolt &symbol=s_it->second;

  code_function_callt call;
  call.function()=symbol_expr(symbol);

  const code_typet::argumentst &arguments=
    to_code_type(symbol.type).arguments();

  call.arguments().resize(
    arguments.size(), static_cast<const exprt &>(get_nil_irep()));

  new_symbol.value=call;
  
  if(context.move(new_symbol))
  {
    messaget message;
    message.set_message_handler(&message_handler);
    message.error("main already defined by another language module");
    return true;
  }
  
  return false;
}

/*******************************************************************\

Function: bp_languaget::parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bp_languaget::parse(
  std::istream &instream,
  const std::string &path,
  message_handlert &message_handler)
{
  bp_parser.clear();

  bp_parser.filename=path;
  bp_parser.in=&instream;
  bp_parser.set_message_handler(&message_handler);

  bool result=bp_parser.parse();

  bp_parse_tree.swap(bp_parser.parse_tree);

  // save some memory
  bp_parser.clear();

  return result;
}

/*******************************************************************\

Function: bp_languaget::modules_provided

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bp_languaget::modules_provided(std::set<std::string> &module_set)
{
}

/*******************************************************************\

Function: bp_languaget::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bp_languaget::typecheck(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  return bp_typecheck(
    bp_parse_tree, context, module, message_handler);
}

/*******************************************************************\

Function: bp_languaget::show_parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
  
void bp_languaget::show_parse(std::ostream &out)
{
  for(bp_parse_treet::declarationst::const_iterator
      it=bp_parse_tree.declarations.begin();
      it!=bp_parse_tree.declarations.end(); it++)
    {
      if(it->id()=="variable")
        out << "VARIABLE: " << *it << std::endl;
      else if(it->id()=="function")
        out << "FUNCTION: " << *it << std::endl;
      else
        out << "UNKNOWN: " << *it << std::endl;

      out << std::endl;
    }
}

/*******************************************************************\

Function: bp_languaget::from_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bp_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  code=expr2bp(expr);
  return false;
}

/*******************************************************************\

Function: bp_languaget::from_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bp_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  code=type2bp(type);
  return false;
}

/*******************************************************************\

Function: bp_languaget::to_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bp_languaget::to_expr(
  const std::string &code,
  const std::string &module,
  exprt &expr,
  message_handlert &message_handler,
  const namespacet &ns)
{
  messaget message(message_handler);
  message.error("not yet implemented");
  return true;
}

/*******************************************************************\

Function: new_bp_language

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
  
languaget *new_bp_language()
{
  return new bp_languaget;
}
