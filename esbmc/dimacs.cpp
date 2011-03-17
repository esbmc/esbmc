/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <fstream>

#include <solvers/sat/dimacs_cnf.h>

#include "bmc.h"
#include "bv_cbmc.h"

/*******************************************************************\

Function: bmc_baset::write_dimacs

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::write_dimacs()
{
  const std::string &filename=options.get_option("outfile");
  
  if(filename.empty() || filename=="-")
    return write_dimacs(std::cout);

  std::ofstream out(filename.c_str());
  if(!out)
  {
    std::cerr << "failed to open " << filename << std::endl;
    return false;
  }

  return write_dimacs(out);
}

/*******************************************************************\

Function: bmc_baset::write_dimacs

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::write_dimacs(std::ostream &out)
{
  dimacs_cnft dimacs_cnf;
  dimacs_cnf.set_message_handler(message_handler);

  bv_cbmct bv_cbmc(dimacs_cnf);

  do_unwind_module(bv_cbmc);
  do_cbmc(bv_cbmc);

  bv_cbmc.dec_solve();

  dimacs_cnf.write_dimacs_cnf(out);
  
  return false;
}
