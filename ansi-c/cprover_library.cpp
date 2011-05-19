/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <sstream>

#include <config.h>

#include "cprover_library.h"
#include "ansi_c_language.h"

/*******************************************************************\

Function: add_cprover_library

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

struct cprover_library_entryt
{
  const char *function;
  const char *model;
} cprover_library[]=
//#include "cprover_library.inc"
{};

void add_cprover_library(
  contextt &context,
  message_handlert &message_handler)
{
  if(config.ansi_c.lib==configt::ansi_ct::LIB_NONE)
    return;

  std::ostringstream library_text;

  library_text <<
    "#line 1 \"<builtin-library>\"\n"
    "#undef inline\n"
    "void __ESBMC_atomic_begin();\n"
    "void __ESBMC_atomic_end();\n"
    "void __ESBMC_yield();\n";

  if(config.ansi_c.string_abstraction)
    library_text << "#define __ESBMC_STRING_ABSTRACTION\n";

  unsigned count=0;

  for(cprover_library_entryt *e=cprover_library;
      e->function!=NULL;
      e++)
  {
    irep_idt id=e->function;

    symbolst::const_iterator old=
      context.symbols.find(id);

    if(old!=context.symbols.end() &&
       old->second.value.is_nil())
    {
      count++;
      library_text << e->model << std::endl;
    }
  }

  if(count>0)
  {
    std::istringstream in(library_text.str());
    ansi_c_languaget ansi_c_language;
    ansi_c_language.parse(in, "", message_handler);

    contextt new_context;
    ansi_c_language.typecheck(
      new_context, "<built-in-library>", message_handler);

    ansi_c_language.merge_context(
      context, new_context,
      message_handler, "<built-in-library>");
  }
}

