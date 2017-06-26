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

#include <cstdint>
#include <esbmc/esbmc_parseoptions.h>
#include <langapi/mode.h>
#include <util/irep2.h>

int main(int argc, const char **argv)
{
  // To avoid the static initialization order fiasco:
  type_poolt bees(true);
  type_pool = bees;

  cbmc_parseoptionst parseoptions(argc, argv);
  return parseoptions.main();
}
