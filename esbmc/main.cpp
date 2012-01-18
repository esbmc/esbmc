/*******************************************************************\

Module: Main Module

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk
		Jeremy Morse, jcmm106@ecs.soton.ac.uk

\*******************************************************************/

/*

  ESBMC
  SMT-based Context-Bounded Model Checking for ANSI-C/C++
  Copyright (c) 2009-2011, Lucas Cordeiro, Federal University of Amazonas
  Jeremy Morse, Denis Nicole, Bernd Fischer, University of Southampton,
  Joao Marques Silva, University College Dublin.
  All rights reserved.

*/

#include <stdint.h>
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
  LANGAPI_HAVE_MODE_CPP,
#ifdef USE_SPECC
  LANGAPI_HAVE_MODE_SPECC,
#endif
#ifdef USE_PHP
  LANGAPI_HAVE_MODE_PHP,
#endif
  LANGAPI_HAVE_MODE_END
};

#if defined(__MINGW32__) && !defined(__MINGW64_VERSION_MAJOR)
extern "C" uint8_t binary___buildidobj_s_start;
uint8_t *version_string = &binary___buildidobj_s_start;
#else
extern "C" uint8_t _binary___buildidobj_s_start;
uint8_t *version_string = &_binary___buildidobj_s_start;
#endif
