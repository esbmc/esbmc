/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_PROP_SMT_DEC_H
#define CPROVER_PROP_SMT_DEC_H

#include <fstream>

#include "smt_conv.h"

class smt_temp_filet
{
public:
  smt_temp_filet();
  ~smt_temp_filet();

protected:  
  std::ofstream temp_out;
  std::string temp_out_filename, temp_result_filename;
};

class smt_dect:protected smt_temp_filet, public smt_convt
{
public:
  smt_dect():smt_convt(temp_out)
  {
  }
  
  virtual resultt dec_solve();
  
protected:
  resultt read_smt_result();
  void read_assert(std::istream &in, std::string &line);
};

#endif
