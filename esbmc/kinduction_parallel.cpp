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

void base_caset::startSolving()
{
  bool res=0;

  // We will do BMC for each k, until MAX_STEPS
  do
  {
    res=_bmc.run(_goto_functions);
    _bmc.options.set_option("unwind", i2string(++_k));
  } while(_k<=MAX_STEPS);
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

void forward_conditiont::startSolving()
{
  bool res=0;

  // We will do BMC for each k, until MAX_STEPS
  do
  {
    res=_bmc.run(_goto_functions);
    _bmc.options.set_option("unwind", i2string(++_k));
  } while(_k<=MAX_STEPS);
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

void inductive_stept::startSolving()
{
  bool res=0;

  // We will do BMC for each k, until MAX_STEPS
  do
  {
    res=_bmc.run(_goto_functions);
    _bmc.options.set_option("unwind", i2string(++_k));
  } while(_k<=MAX_STEPS);
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
  :  bc_finished(false),
     fc_finished(false),
     is_finished(false)
{
  pthread_mutex_init(&_mutex, NULL);

  for(unsigned i=0; i<MAX_STEPS; ++i)
  {
    bc_queue[i]=MAX_STEPS+1;
    fc_queue[i]=MAX_STEPS+1;
    is_queue[i]=MAX_STEPS+1;
  }
}

void safe_queues::update_queue(STEP s, unsigned int k, int res)
{
  pthread_mutex_lock(&_mutex);

  switch(s)
  {
    case BASE_CASE:
      bc_queue[k]=res;
      break;

    case FORWARD_CONDITION:
      fc_queue[k]=res;
      break;

    case INDUCTIVE_STEP:
      is_queue[k]=res;
      break;
  }

  // Check if a solution was found
  if(bc_queue[k] != MAX_STEPS+1)
  {
    if(bc_queue[k])
    {
//      finalize_multithread();
    }

    if(fc_queue[k] != MAX_STEPS+1)
    {
      if(!fc_queue[k])
      {
//        finalize_multithread();
      }
    }

    if(is_queue[k] != MAX_STEPS+1)
    {
      if(!is_queue[k])
      {
//        finalize_multithread();
      }
    }
  }

  pthread_mutex_unlock(&_mutex);
}

void safe_queues::update_finished(STEP s, bool finished)
{
  switch(s)
  {
    case BASE_CASE:
      bc_finished=finished;
      break;

    case FORWARD_CONDITION:
      fc_finished=finished;
      break;

    case INDUCTIVE_STEP:
      is_finished=finished;
      break;
  }

  if(bc_finished  && fc_finished  && is_finished)
  {
//    pthread_mutex_lock(&main_mutex);
//    pthread_cond_broadcast(&main_cond);
//    pthread_mutex_unlock(&main_mutex);
  }

}
