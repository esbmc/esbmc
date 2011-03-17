/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <fstream>

#include <solvers/z3/z3_dec.h>

#include "bmc.h"
#include "bv_cbmc.h"

/*******************************************************************\

Function: bmc_baset::z3

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::z3()
{
  return decide_solver_z3();
}

/*******************************************************************\

Function: bmc_baset::z3_conv

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::z3_conv(std::ostream &out)
{
  z3_convt z3_conv(out);
  z3_conv.set_message_handler(message_handler);

  do_unwind_module(z3_conv);
  do_cbmc(z3_conv);
  z3_conv.dec_solve();

  return false;
}
