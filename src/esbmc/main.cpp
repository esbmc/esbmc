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

#include <irep2.h>

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
