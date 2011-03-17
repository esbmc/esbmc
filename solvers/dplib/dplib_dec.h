/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_PROP_DPLIB_DEC_H
#define CPROVER_PROP_DPLIB_DEC_H

#include <fstream>

#include "dplib_conv.h"

class dplib_temp_filet
{
public:
  dplib_temp_filet();
  ~dplib_temp_filet();

protected:  
  std::ofstream temp_out;
  std::string temp_out_filename, temp_result_filename;
};

class dplib_dect:protected dplib_temp_filet, public dplib_convt
{
public:
  dplib_dect():dplib_convt(temp_out)
  {
  }
  
  virtual resultt dec_solve();
  
protected:
  resultt read_dplib_result();
  void read_assert(std::istream &in, std::string &line);
};

#endif
