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

resultt base_caset::startSolving()
{
  resultt r;
  r.step=BASE_CASE;
  r.k=_k;
  r.result=0;

  r.result=_bmc.run(_goto_functions);
  _bmc.options.set_option("unwind", i2string(++_k));

  return r;
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

resultt forward_conditiont::startSolving()
{

  resultt r;
  r.step=FORWARD_CONDITION;
  r.k=_k;
  r.result=1;

  r.result=_bmc.run(_goto_functions);
  _bmc.options.set_option("unwind", i2string(++_k));

  return r;
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

resultt inductive_stept::startSolving()
{

  resultt r;
  r.step=INDUCTIVE_STEP;
  r.k=_k;
  r.result=1;

  r.result=_bmc.run(_goto_functions);
  _bmc.options.set_option("unwind", i2string(++_k));

  return r;
}
