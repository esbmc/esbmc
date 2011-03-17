/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <fstream>

#include <solvers/boolector/boolector_dec.h>

#include "bmc.h"
#include "bv_cbmc.h"

/*******************************************************************\

Function: bmc_baset::boolector

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::boolector()
{
  boolector_dect boolector_dec;
  decide_solver_boolector();
  return true;
}

/*******************************************************************\

Function: bmc_baset::boolector_conv

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::boolector_conv(std::ostream &out)
{
  boolector_convt boolector_conv(out);
  boolector_conv.set_message_handler(message_handler);

  do_unwind_module(boolector_conv);
  do_cbmc(boolector_conv);

  boolector_conv.dec_solve();

  return false;
}
