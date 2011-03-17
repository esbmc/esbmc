/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <sstream>
#include <fstream>

#include <ansi-c/c_preprocess.h>
#include <ansi-c/c_final.h>

#include "promela_language.h"
#include "expr2promela.h"
#include "promela_parser.h"
#include "promela_typecheck.h"

#include <ansi-c/c_link.h>
#include <ansi-c/c_sequent.h>

/*******************************************************************\

Function: promela_languaget::modules_provided

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void promela_languaget::modules_provided(std::set<std::string> &modules)
{
  modules.insert(parse_path);
}

/*******************************************************************\

Function: internal_additions

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void internal_additions(std::string &code)
{
  code+=
    "void assume(bool assumption);\n"
    "void assert(bool assertion);\n"
    "\n\n";
}

/*******************************************************************\

Function: promela_languaget::preprocess

  Inputs:

 Outputs:

 Purpose: ANSI-C preprocessing

\*******************************************************************/

bool promela_languaget::preprocess(std::istream &instream,
                                 const std::string &path,
                                 std::ostream &outstream,
                                 std::ostream &err)
{
  return c_preprocess(instream, path, outstream, err);
}
             
/*******************************************************************\

Function: promela_languaget::parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool promela_languaget::parse(std::istream &instream,
                            const std::string &path,
                            std::ostream &err)
{
  // store the path

  parse_path=path;

  // preprocessing

  std::ostringstream o_preprocessed;

  if(preprocess(instream, path, o_preprocessed, err))
    return true;

  std::istringstream i_preprocessed(o_preprocessed.str());

  // parsing

  std::string code;
  internal_additions(code);
  std::istringstream codestr(code);

  promela_parser.clear();
  promela_parser.filename=path;
  promela_parser.in=&codestr;
  promela_parser.err=&err;
  promela_parser.grammar=promela_parsert::LANGUAGE;
  promela_scanner_init();

  bool result=promela_parser.parse();

  if(!result)
  {
    promela_parser.in=&i_preprocessed;
    promela_scanner_init();
    result=promela_parser.parse();
  }

  // save result
  promela_parse_tree.swap(promela_parser.parse_tree);

  // save some memory
  promela_parser.clear();

  return result;
}
             
/*******************************************************************\

Function: promela_languaget::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool promela_languaget::typecheck(contextt &context,
                                const std::string &module,
                                std::ostream &err)
{
  contextt new_context;

  if(promela_typecheck(promela_parse_tree, new_context, module, err))
    return true;

  return c_link(context, new_context, err, module);
}

/*******************************************************************\

Function: promela_languaget::final

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool promela_languaget::final(contextt &context, std::ostream &err)
{
  return c_final(context, err);
}

/*******************************************************************\

Function: promela_languaget::make_sequent

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void promela_languaget::make_sequent(const contextt &context,
                                   const symbolt &symbol,
                                   h_sequentt &sequent)
{
  if(symbol.type.id()!="code")
  {
    languaget::make_sequent(context, symbol, sequent);
    return;
  }

  c_make_sequent(context, symbol, sequent,
                 symbol.name=="promela::main");
}

/*******************************************************************\

Function: promela_languaget::show_parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
  
void promela_languaget::show_parse(std::ostream &out)
{
  for(promela_parse_treet::declarationst::const_iterator it=
      promela_parse_tree.declarations.begin();
      it!=promela_parse_tree.declarations.end();
      it++)
  {
    out << it->get("scope") << it->get("name") << ":"
        << it->type() << "="
        << it->find("value") << std::endl;
  }
}

/*******************************************************************\

Function: new_promela_language

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

languaget *new_promela_language()
{
  return new promela_languaget;
}

/*******************************************************************\

Function: promela_languaget::from_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool promela_languaget::from_expr(const exprt &expr, std::string &code,
                                const namespacet &ns)
{
  return expr2promela(expr, ns, code);
}

/*******************************************************************\

Function: promela_languaget::from_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool promela_languaget::from_type(const typet &type, std::string &code,
                                const namespacet &ns)
{
  return type2promela(type, ns, code);
}

/*******************************************************************\

Function: promela_languaget::to_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
                         
bool promela_languaget::to_expr(const std::string &code,
                                const std::string &module,
                                exprt &expr,
                                std::ostream &err,
                                const namespacet &ns)
{
  expr.make_nil();

  // no preprocessing yet...

  std::istringstream i_preprocessed(code);

  // parsing

  promela_parser.clear();
  promela_parser.filename="";
  promela_parser.in=&i_preprocessed;
  promela_parser.err=&err;
  promela_parser.grammar=promela_parsert::EXPRESSION;
  promela_scanner_init();

  bool result=promela_parser.parse();

  #if 0
  if(promela_parser.parse.declarations.empty())
    result=true;
  else
  {
    expr.swap(promela_parser.parse.declarations.front());

    // typecheck it
    result=promela_typecheck(expr, err, ns);
  }
  #endif

  // save some memory
  promela_parser.clear();

  return result;
}

/*******************************************************************\

Function: promela_languaget::~promela_languaget

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

promela_languaget::~promela_languaget()
{
}
