/*******************************************************************\

Module: Main Module

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

/*

  CBMC
  Bounded Model Checking for ANSI-C
  Copyright (C) 2001-2005 Daniel Kroening <kroening@kroening.com>

*/

#include <langapi/mode.h>

#include "parseoptions.h"

/*******************************************************************\

Function: main

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int main(int argc, const char **argv)
{
  cbmc_parseoptionst parseoptions(argc, argv);
  return parseoptions.main();
}

const mode_table_et mode_table[] =
{
  LANGAPI_HAVE_MODE_C,
  LANGAPI_HAVE_MODE_CVC,
#ifdef USE_CPP
  LANGAPI_HAVE_MODE_CPP,
#endif
#ifdef USE_SPECC
  LANGAPI_HAVE_MODE_SPECC,
#endif
#ifdef USE_PHP
  LANGAPI_HAVE_MODE_PHP,
#endif
  LANGAPI_HAVE_MODE_END
};
