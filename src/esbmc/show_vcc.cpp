/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <esbmc/bmc.h>
#include <fstream>
#include <iostream>
#include <langapi/language_util.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <util/migrate.h>

/*******************************************************************\

Function: bmct::show_vcc

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmct::show_vcc(std::ostream &out, symex_target_equationt &equation)
{
  switch(ui)
  {
  case ui_message_handlert::OLD_GUI:
  case ui_message_handlert::XML_UI:
    error("not supported");
    return;

  case ui_message_handlert::PLAIN:
    break;

  default:
    assert(false);
  }

  out << std::endl << "VERIFICATION CONDITIONS:" << std::endl << std::endl;

  languagest languages(ns, MODE_C);

  for(symex_target_equationt::SSA_stepst::iterator
      it=equation.SSA_steps.begin();
      it!=equation.SSA_steps.end(); it++)
  {
    if(!it->is_assert()) continue;

    if(it->source.pc->location.is_not_nil())
      out << it->source.pc->location << std::endl;

    if(it->comment!="")
      out << it->comment << std::endl;

    symex_target_equationt::SSA_stepst::const_iterator
      p_it=equation.SSA_steps.begin();

    for(unsigned count=1; p_it!=it; p_it++)
      if(p_it->is_assume() || p_it->is_assignment())
        if(!p_it->ignore)
        {
          std::string string_value;
          languages.from_expr(migrate_expr_back(p_it->cond), string_value);
          out << "{-" << count << "} " << string_value << std::endl;
          count++;
        }

    out << "|--------------------------" << std::endl;

    std::string string_value;
    languages.from_expr(migrate_expr_back(it->cond), string_value);
    out << "{" << 1 << "} " << string_value << std::endl;

    out << std::endl;
  }
}

/*******************************************************************\

Function: bmct::show_vcc

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmct::show_vcc(symex_target_equationt &equation)
{
  const std::string &filename=options.get_option("output");

  if(filename.empty() || filename=="-")
    show_vcc(std::cout, equation);
  else
  {
    std::ofstream out(filename.c_str());
    if(!out)
      std::cerr << "failed to open " << filename << std::endl;
    else
      show_vcc(out, equation);
  }
}

