/*
 * K_induction_parallel.cpp
 *
 *  Created on: 27/06/2013
 *      Author: ceteli04
 */

#include "kinduction_parallel.h"

kinduction_thread::kinduction_thread(bmct bmc, goto_functionst goto_functions)
  : Thread(),
    k(0),
    _bmc(bmc),
    _goto_functions(goto_functions)
{ }

/* Base case class implementation */

base_case_thread::base_case_thread(bmct bmc, goto_functionst goto_functions)
  : kinduction_thread(bmc, goto_functions)
{ }

void base_case_thread::run()
{

}

/* Forward condition class implementation */

forward_condition_thread::forward_condition_thread(bmct bmc, goto_functionst goto_functions)
  : kinduction_thread(bmc, goto_functions)
{ }

void forward_condition_thread::run()
{

}

/* Inductive step class implementation */

inductive_step_thread::inductive_step_thread(bmct bmc, goto_functionst goto_functions)
  : kinduction_thread(bmc, goto_functions)
{ }

void inductive_step_thread::run()
{

}

/* class safe_queues */
safe_queues *safe_queues::instance=NULL;

safe_queues *safe_queues::get_instance()
{
  if(!instance)
    instance=new safe_queues;
  return instance;
}

safe_queues::safe_queues()
  : _bcMutex(PTHREAD_MUTEX_INITIALIZER),
    _fcMutex(PTHREAD_MUTEX_INITIALIZER),
    _isMutex(PTHREAD_MUTEX_INITIALIZER)
{
  for(unsigned i=0; i<50; ++i)
  {
    bc_queue[i]=-1;
    fc_queue[i]=-1;
    is_queue[i]=-1;
  }
}
