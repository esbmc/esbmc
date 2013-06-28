/*
 * K_induction_parallel.cpp
 *
 *  Created on: 27/06/2013
 *      Author: ceteli04
 */

#include "kinduction_parallel.h"

pthread_mutex_t main_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t main_cond = PTHREAD_COND_INITIALIZER;

pthread_mutex_t solution_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t solution_cond = PTHREAD_COND_INITIALIZER;

bool solution_found=false;

// This function will unlock the main thread and set solution found
// to true
void finalize_multithread()
{
  pthread_mutex_lock(&solution_mutex);
  solution_found=true;
  pthread_mutex_unlock(&solution_mutex);

  pthread_mutex_lock(&main_mutex);
  pthread_cond_signal(&main_cond);
  pthread_mutex_unlock(&main_mutex);
}

/* Base case class implementation */

base_case_thread::base_case_thread(bmct &bmc,
    goto_functionst &goto_functions)
  : Thread(),
    _k(1),
    _bmc(bmc),
    _goto_functions(goto_functions)
{
  _bmc.options.set_option("unwind", i2string(_k));
}

void base_case_thread::run()
{
  bool res=0;
  bool found_solution=false;

  // We will do BMC for each k, until 50
  do
  {
    res=_bmc.run(_goto_functions);

    // If the base case is false, the property is false
    if(!res)
    {
      finalize_multithread();
      return;
    }

    safe_queues::get_instance()->update_bc_queue(_k,res);

    _bmc.options.set_option("unwind", i2string(++_k));

    pthread_mutex_lock(&solution_mutex);
    found_solution=solution_found;
    pthread_mutex_unlock(&solution_mutex);

  } while(_k<=50 && found_solution);
}

/* Forward condition class implementation */

forward_condition_thread::forward_condition_thread(bmct &bmc,
    goto_functionst &goto_functions)
  : Thread(),
    _k(2),
    _bmc(bmc),
    _goto_functions(goto_functions)
{
  _bmc.options.set_option("unwind", i2string(_k));
}


void forward_condition_thread::run()
{
  bool res=0;
  bool found_solution=false;

  // We will do BMC for each k, until 50
  do
  {
    res=_bmc.run(_goto_functions);
    safe_queues::get_instance()->update_fc_queue(_k,res);

    _bmc.options.set_option("unwind", i2string(++_k));

    pthread_mutex_lock(&solution_mutex);
    found_solution=solution_found;
    pthread_mutex_unlock(&solution_mutex);

  } while(_k<=50 && found_solution);
}

/* Inductive step class implementation */

inductive_step_thread::inductive_step_thread(bmct &bmc,
    goto_functionst &goto_functions)
  : Thread(),
    _k(2),
    _bmc(bmc),
    _goto_functions(goto_functions)
{
  _bmc.options.set_option("unwind", i2string(_k));
}


void inductive_step_thread::run()
{
  bool res=0;
  bool found_solution=false;

  // We will do BMC for each k, until 50
  do
  {
    res=_bmc.run(_goto_functions);
    safe_queues::get_instance()->update_is_queue(_k,res);

    _bmc.options.set_option("unwind", i2string(++_k));

    pthread_mutex_lock(&solution_mutex);
    found_solution=solution_found;
    pthread_mutex_unlock(&solution_mutex);

  } while(_k<=50 && found_solution);
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
{
  pthread_mutex_init(&_bcMutex, NULL);
  pthread_mutex_init(&_fcMutex, NULL);
  pthread_mutex_init(&_isMutex, NULL);

  for(unsigned i=0; i<50; ++i)
  {
    bc_queue[i]=-1;
    fc_queue[i]=-1;
    is_queue[i]=-1;
  }
}

void safe_queues::update_bc_queue(unsigned int k, int res)
{
  pthread_mutex_lock(&_bcMutex);
  bc_queue[k]=res;
  pthread_mutex_unlock(&_bcMutex);
}

void safe_queues::update_fc_queue(unsigned int k, int res)
{
  pthread_mutex_lock(&_fcMutex);
  fc_queue[k]=res;
  pthread_mutex_unlock(&_fcMutex);
}

void safe_queues::update_is_queue(unsigned int k, int res)
{
  pthread_mutex_lock(&_isMutex);
  is_queue[k]=res;
  pthread_mutex_unlock(&_isMutex);
}
