/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef CPROVER_PROP_BOOLECTOR_DEC_H
#define CPROVER_PROP_BOOLECTOR_DEC_H

#include <fstream>

#include "boolector_conv.h"

class boolector_temp_filet
{
public:
  boolector_temp_filet();
  ~boolector_temp_filet();

protected:
  std::ofstream temp_out;
  std::string temp_out_filename, temp_result_filename;
};

class boolector_dect:protected boolector_temp_filet, public boolector_convt
{
public:
  boolector_dect():boolector_convt(temp_out)
  {
  }

  virtual resultt dec_solve();

  void set_file(std::string file);
  void set_btor(bool btor);

protected:
  resultt read_boolector_result();
  void read_assert(std::istream &in, std::string &line);

private:
  bool btor_lang;
};

#endif
