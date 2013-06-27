/*
 * K_induction_parallel.cpp
 *
 *  Created on: 27/06/2013
 *      Author: ceteli04
 */

#include "kinduction_parallel.h"

kinduction_thread::kinduction_thread()
  : Thread(),
    k(0)
{ }

/* Base case class implementation */

base_case_thread::base_case_thread()
  : kinduction_thread()
{

}

void base_case_thread::run()
{

}

/* Forward condition class implementation */

forward_condition_thread::forward_condition_thread()
  : kinduction_thread()
{

}

void forward_condition_thread::run()
{

}

/* Inductive step class implementation */

inductive_step_thread::inductive_step_thread()
  : kinduction_thread()
{

}

void inductive_step_thread::run()
{

}
