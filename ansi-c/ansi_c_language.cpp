/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <string.h>

#include <sstream>
#include <fstream>

#include <config.h>
#include <expr_util.h>

#include "ansi_c_language.h"
#include "ansi_c_convert.h"
#include "ansi_c_typecheck.h"
#include "ansi_c_parser.h"
#include "expr2c.h"
#include "c_final.h"
#include "trans_unit.h"
#include "c_link.h"
#include "c_preprocess.h"
#include "c_main.h"
#include "gcc_builtin_headers.h"

/*******************************************************************\

Function: ansi_c_languaget::modules_provided

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_languaget::modules_provided(std::set<std::string> &modules)
{
  modules.insert(translation_unit(parse_path));
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
    "void __ESBMC_assume(_Bool assumption);\n"
    "void assert(_Bool assertion);\n"
    "void __ESBMC_assert(_Bool assertion, const char *description);\n"
    "_Bool __ESBMC_same_object(const void *, const void *);\n"
    "_Bool __ESBMC_is_zero_string(const void *);\n"
    "unsigned __ESBMC_zero_string_length(const void *);\n"
    "unsigned __ESBMC_buffer_size(const void *);\n"

    // traces
    "void CBMC_trace(int lvl, const char* event, ...);\n"

    // pointers
    "unsigned __ESBMC_POINTER_OBJECT(const void *p);\n"
    "signed __ESBMC_POINTER_OFFSET(const void *p);\n"

    // malloc
    "unsigned __ESBMC_constant_infinity_uint;\n"
    "_Bool __ESBMC_alloc[__ESBMC_constant_infinity_uint];\n"
    "unsigned __ESBMC_alloc_size[__ESBMC_constant_infinity_uint];\n"

    // this is ANSI-C
    "extern const char __func__[];\n"

    // float stuff
    "_Bool __ESBMC_isnan(double f);\n"
    "_Bool __ESBMC_isfinite(double f);\n"
    "_Bool __ESBMC_isinf(double f);\n"
    "_Bool __ESBMC_isnormal(double f);\n"
    "extern int __ESBMC_rounding_mode;\n"

    // absolute value
    "int __ESBMC_abs(int x);\n"
    "long int __ESBMC_labs(long int x);\n"
    "double __ESBMC_fabs(double x);\n"
    "long double __ESBMC_fabsl(long double x);\n"
    "float __ESBMC_fabsf(float x);\n"

    // GCC junk stuff
    GCC_BUILTIN_HEADERS

    "\n";
}

/*******************************************************************\

Function: ansi_c_languaget::preprocess

  Inputs:

 Outputs:

 Purpose: ANSI-C preprocessing

\*******************************************************************/

bool ansi_c_languaget::preprocess(
  std::istream &instream,
  const std::string &path,
  std::ostream &outstream,
  message_handlert &message_handler)
{
  // check extensions

  const char *ext=strrchr(path.c_str(), '.');
  if(ext!=NULL && std::string(ext)==".i")
  {
    std::ifstream infile(path.c_str());

    char ch;

    while(infile.read(&ch, 1))
      outstream << ch;

    return false;
  }

  return c_preprocess(instream, path, outstream, message_handler);
}

/*******************************************************************\

Function: ansi_c_languaget::parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ansi_c_languaget::parse(
  std::istream &instream,
  const std::string &path,
  message_handlert &message_handler)
{
  // store the path

  parse_path=path;

  // preprocessing

  std::ostringstream o_preprocessed;

  if(preprocess(instream, path, o_preprocessed, message_handler))
    return true;

  std::istringstream i_preprocessed(o_preprocessed.str());

  // parsing

  std::string code;
  internal_additions(code);
  std::istringstream codestr(code);

  ansi_c_parser.clear();
  ansi_c_parser.filename="<built-in>";
  ansi_c_parser.in=&codestr;
  ansi_c_parser.set_message_handler(&message_handler);
  ansi_c_parser.grammar=ansi_c_parsert::LANGUAGE;

  if(config.ansi_c.os==configt::ansi_ct::OS_WIN32)
    ansi_c_parser.mode=ansi_c_parsert::MSC;
  else
    ansi_c_parser.mode=ansi_c_parsert::GCC;

  ansi_c_scanner_init();

  bool result=ansi_c_parser.parse();

  if(!result)
  {
    ansi_c_parser.line_no=0;
    ansi_c_parser.filename=path;
    ansi_c_parser.in=&i_preprocessed;
    ansi_c_scanner_init();
    result=ansi_c_parser.parse();
  }

  // save result
  parse_tree.swap(ansi_c_parser.parse_tree);

  // save some memory
  ansi_c_parser.clear();

  return result;
}

/*******************************************************************\

Function: ansi_c_languaget::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ansi_c_languaget::typecheck(
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  if(ansi_c_convert(parse_tree, module, message_handler))
    return true;

  contextt new_context;

  if(ansi_c_typecheck(parse_tree, new_context, module, message_handler))
    return true;

  if(c_link(context, new_context, message_handler, module))
    return true;

  return false;
}

/*******************************************************************\

Function: ansi_c_languaget::final

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ansi_c_languaget::final(
  contextt &context,
  message_handlert &message_handler)
{
  if(c_final(context, message_handler)) return true;
  if(c_main(context, "c::", "c::main", message_handler)) return true;

  return false;
}

/*******************************************************************\

Function: ansi_c_languaget::show_parse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ansi_c_languaget::show_parse(std::ostream &out)
{
  parse_tree.output(out);
}

/*******************************************************************\

Function: new_ansi_c_language

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

languaget *new_ansi_c_language()
{
  return new ansi_c_languaget;
}

/*******************************************************************\

Function: ansi_c_languaget::from_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ansi_c_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  code=expr2c(expr, ns);
  return false;
}

/*******************************************************************\

Function: ansi_c_languaget::from_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ansi_c_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  code=type2c(type, ns);
  return false;
}

/*******************************************************************\

Function: ansi_c_languaget::to_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ansi_c_languaget::to_expr(
  const std::string &code __attribute__((unused)),
  const std::string &module __attribute__((unused)),
  exprt &expr,
  message_handlert &message_handler,
  const namespacet &ns)
{
  expr.make_nil();

  // no preprocessing yet...

  std::istringstream i_preprocessed(code);

  // parsing

  ansi_c_parser.clear();
  ansi_c_parser.filename="";
  ansi_c_parser.in=&i_preprocessed;
  ansi_c_parser.set_message_handler(&message_handler);
  ansi_c_parser.grammar=ansi_c_parsert::EXPRESSION;
  ansi_c_parser.mode=ansi_c_parsert::GCC;
  ansi_c_scanner_init();

  bool result=ansi_c_parser.parse();

  if(ansi_c_parser.parse_tree.declarations.empty())
    result=true;
  else
  {
    expr.swap(ansi_c_parser.parse_tree.declarations.front());

    result=ansi_c_convert(expr, "", message_handler);

    // typecheck it
    if(!result)
      result=ansi_c_typecheck(expr, message_handler, ns);
  }

  // save some memory
  ansi_c_parser.clear();

  return result;
}

/*******************************************************************\

Function: ansi_c_languaget::~ansi_c_languaget

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

ansi_c_languaget::~ansi_c_languaget()
{
}

/*******************************************************************\

Function: ansi_c_languaget::merge_context

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ansi_c_languaget::merge_context(
  contextt &dest,
  contextt &src,
  message_handlert &message_handler,
  const std::string &module) const
{
  return c_link(dest, src, message_handler, module);
}
