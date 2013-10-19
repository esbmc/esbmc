/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <iostream>

#if defined (_WIN32)
#define EX_OK 0
#define EX_USAGE 1
#else
#include <sysexits.h>
#endif

#include "cmdline.h"
#include "parseoptions.h"
#include "signal_catcher.h"

/*******************************************************************\

Function: parseoptions_baset::parseoptions_baset

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

parseoptions_baset::parseoptions_baset(
  const struct opt_templ *opts, int argc, const char **argv)
{
  parse_result = cmdline.parse(argc, argv, opts);
}

/*******************************************************************\

Function: parseoptions_baset::help

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void parseoptions_baset::help()
{
}

/*******************************************************************\

Function: parseoptions_baset::main

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int parseoptions_baset::main()
{
  if(parse_result)
  {
    std::cerr << "Unrecognized option \"" << cmdline.failing_option << "\"";
    std::cerr << std::endl;
    return EX_USAGE;
  }

  if(cmdline.isset('?') || cmdline.isset('h') || cmdline.isset("help"))
  {
    help();
    return EX_OK;
  }

  // install signal catcher
  install_signal_catcher();

  if(cmdline.isset("k-induction"))
    return doit_k_induction();

  return doit();
}
