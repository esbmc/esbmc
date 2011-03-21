/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "smv_language.h"
#include "smv_typecheck.h"
#include "smv_parser.h"
#include "expr2smv.h"

/*******************************************************************\

Function: smv_languaget::parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smv_languaget::parse(
  std::istream &instream,
  const std::string &path,
  message_handlert &message_handler)
{
  smv_parser.clear();

  const std::string main_name=smv_module_symbol("main");
  smv_parser.module=&smv_parser.parse_tree.modules[main_name];
  smv_parser.module->name=main_name;
  smv_parser.module->base_name="main";

  smv_parser.filename=path;
  smv_parser.in=&instream;
  smv_parser.set_message_handler(&message_handler);

  bool result=smv_parser.parse();

  // see if we used main

  if(!smv_parser.parse_tree.modules[main_name].used)
    smv_parser.parse_tree.modules.erase(main_name);

  smv_parse_tree.swap(smv_parser.parse_tree);

  // save some memory
  smv_parser.clear();

  return result;
}

/*******************************************************************\

Function: smv_languaget::dependencies

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smv_languaget::dependencies(
  const std::string &module, 
  std::set<std::string> &module_set)
{
  smv_parse_treet::modulest::const_iterator
    m_it=smv_parse_tree.modules.find(module);

  if(m_it==smv_parse_tree.modules.end()) return;

  const smv_parse_treet::modulet &smv_module=m_it->second;

  for(smv_parse_treet::mc_varst::const_iterator it=smv_module.vars.begin();
      it!=smv_module.vars.end(); it++)
    if(it->second.type.id()=="submodule")
      module_set.insert(it->second.type.get_string("identifier"));
}

/*******************************************************************\

Function: smv_languaget::modules_provided

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void smv_languaget::modules_provided(std::set<std::string> &module_set)
{
  for(smv_parse_treet::modulest::const_iterator
      it=smv_parse_tree.modules.begin();
      it!=smv_parse_tree.modules.end(); it++)
    module_set.insert(id2string(it->second.name));
}

/*******************************************************************\

Function: smv_languaget::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smv_languaget::typecheck(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  return smv_typecheck(smv_parse_tree, context, module, message_handler);
}

/*******************************************************************\

Function: smv_languaget::show_parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
  
void smv_languaget::show_parse(std::ostream &out)
{
  for(smv_parse_treet::modulest::const_iterator
      it=smv_parse_tree.modules.begin();
      it!=smv_parse_tree.modules.end(); it++)
  {
    const smv_parse_treet::modulet &module=it->second;
    out << "Module: " << module.name << std::endl << std::endl;

    out << "  VARIABLES:" << std::endl;

    for(smv_parse_treet::mc_varst::const_iterator it=module.vars.begin();
        it!=module.vars.end(); it++)
      if(it->second.type.id()!="submodule")
      {
        std::string msg;
        type2smv(it->second.type, msg);
        out << "    " << it->first << ": " 
            << msg << ";" << std::endl;
      }

    out << std::endl;

    out << "  SUBMODULES:" << std::endl;

    for(smv_parse_treet::mc_varst::const_iterator
        it=module.vars.begin();
        it!=module.vars.end(); it++)
      if(it->second.type.id()=="submodule")
      {
        std::string msg;
        type2smv(it->second.type, msg);
        out << "    " << it->first << ": " 
            << msg << ";" << std::endl;
      }

    out << std::endl;

    out << "  ITEMS:" << std::endl;

    forall_item_list(it, module.items)
    {
      out << "    TYPE: " << to_string(it->item_type) << std::endl;
      out << "    EXPR: " << it->expr << std::endl;
      out << std::endl;
    }
  }
}

/*******************************************************************\

Function: smv_languaget::from_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smv_languaget::from_expr(const exprt &expr, std::string &code,
                              const namespacet &ns __attribute__((unused)))
{
  return expr2smv(expr, code);
}

/*******************************************************************\

Function: smv_languaget::from_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smv_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns __attribute__((unused)))
{
  return type2smv(type, code);
}

/*******************************************************************\

Function: smv_languaget::to_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool smv_languaget::to_expr(
  const std::string &code __attribute__((unused)),
  const std::string &module __attribute__((unused)),
  exprt &expr __attribute__((unused)),
  message_handlert &message_handler,
  const namespacet &ns __attribute__((unused)))
{
  messaget message(message_handler);
  message.error("not yet implemented");
  return true;
}

/*******************************************************************\

Function: new_smv_language

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
  
languaget *new_smv_language()
{
  return new smv_languaget;
}

