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

base_case::base_case()
  : kinduction_thread()
{

}

void base_case::run()
{

}

/* Forward condition class implementation */

forward_condition::forward_condition()
  : kinduction_thread()
{

}

void forward_condition::run()
{

}

/* Inductive step class implementation */

inductive_step::inductive_step()
  : kinduction_thread()
{

}

void inductive_step::run()
{

}
