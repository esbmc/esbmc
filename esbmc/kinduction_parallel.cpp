/*
 * K_induction_parallel.cpp
 *
 *  Created on: 27/06/2013
 *      Author: ceteli04
 */

#include "kinduction_parallel.h"

/* Base case class implementation */

base_caset::base_caset(bmct &bmc,
    goto_functionst &goto_functions)
  : _k(1),
    _bmc(bmc),
    _goto_functions(goto_functions)
{
  _bmc.options.set_option("unwind", i2string(_k));
}

bool base_caset::startSolving()
{
  bool res=0;

  // We will do BMC for each k, until MAX_STEPS
  do
  {
    res=_bmc.run(_goto_functions);

    // If it fails, a bug was found
    if(res) return res;

    _bmc.options.set_option("unwind", i2string(++_k));
  } while(_k<=MAX_STEPS);

  return false;
}

/* Forward condition class implementation */

forward_conditiont::forward_conditiont(bmct &bmc,
    goto_functionst &goto_functions)
  : _k(2),
    _bmc(bmc),
    _goto_functions(goto_functions)
{
  _bmc.options.set_option("unwind", i2string(_k));
}

bool forward_conditiont::startSolving()
{
  bool res=0;

  // We will do BMC for each k, until MAX_STEPS
  do
  {
    res=_bmc.run(_goto_functions);

    // If this was a success, the property was proved
    if(!res) return res;

    _bmc.options.set_option("unwind", i2string(++_k));
  } while(_k<=MAX_STEPS);

  return true;
}

/* Inductive step class implementation */

inductive_stept::inductive_stept(bmct &bmc,
    goto_functionst &goto_functions)
  : _k(2),
    _bmc(bmc),
    _goto_functions(goto_functions)
{
  _bmc.options.set_option("unwind", i2string(_k));
}

bool inductive_stept::startSolving()
{
  bool res=0;

  // We will do BMC for each k, until MAX_STEPS
  do
  {
    res=_bmc.run(_goto_functions);

    // If this was a success, the property was proved
    if(!res) return res;

    _bmc.options.set_option("unwind", i2string(++_k));
  } while(_k<=MAX_STEPS);

  return true;
}
