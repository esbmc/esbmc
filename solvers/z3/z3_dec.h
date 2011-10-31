/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef CPROVER_PROP_z3_DEC_H
#define CPROVER_PROP_z3_DEC_H

#include <fstream>

#include "z3_conv.h"

class z3_temp_filet
{
public:
  z3_temp_filet();
  ~z3_temp_filet();

protected:
  std::ofstream temp_out;
  std::string temp_out_filename, temp_result_filename;
};

class z3_dect:protected z3_temp_filet, public z3_convt
{
public:
  z3_dect(bool rel, bool uw, bool int_encoding, bool smtlib)
    :z3_convt(temp_out, rel, uw, int_encoding, smtlib)
  {
    this->smtlib = smtlib;
  }

  virtual resultt dec_solve();
  void set_file(std::string file);
  void set_ecp(bool ecp);
  bool get_unsat_core(void);
  bool get_number_of_assumptions(void);
  void set_unsat_core(uint val);

protected:
  resultt read_z3_result();
  void read_assert(std::istream &in, std::string &line);

private:
  bool smtlib;

};

#endif
