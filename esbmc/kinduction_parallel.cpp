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

pthread_mutex_t kill_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t kill_cond = PTHREAD_COND_INITIALIZER;

// This function will unlock the main thread and set solution found
// to true
void finalize_multithread()
{
  pthread_mutex_lock(&solution_mutex);
  solution_found=true;
  pthread_mutex_unlock(&solution_mutex);
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

base_case_thread::~base_case_thread()
{
//  std::cout << "### BC Dying" << std::endl;
}

void base_case_thread::run()
{
  bool res=0;
  bool found_solution=false;

  // We will do BMC for each k, until MAX_STEPS
  do
  {
    res=_bmc.run(_goto_functions);

    safe_queues::get_instance()->update_queue(safe_queues::BASE_CASE, _k, res);

    _bmc.options.set_option("unwind", i2string(++_k));

    pthread_mutex_lock(&solution_mutex);
    found_solution=solution_found;
    pthread_mutex_unlock(&solution_mutex);

  } while(_k<=MAX_STEPS && !found_solution);

  safe_queues::get_instance()->update_finished(safe_queues::BASE_CASE, true);

//  std::cout << "### Stopping BC" << std::endl;

  pthread_mutex_lock(&kill_mutex);
  pthread_cond_wait(&kill_cond, &kill_mutex);
  pthread_mutex_unlock(&kill_mutex);
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

forward_condition_thread::~forward_condition_thread()
{
//  std::cout << "### FC Dying" << std::endl;
}

void forward_condition_thread::run()
{
  bool res=0;
  bool found_solution=false;

  // We will do BMC for each k, until MAX_STEPS
  do
  {
    res=_bmc.run(_goto_functions);
    safe_queues::get_instance()->update_queue(safe_queues::FORWARD_CONDITION, _k, res);

    _bmc.options.set_option("unwind", i2string(++_k));

    pthread_mutex_lock(&solution_mutex);
    found_solution=solution_found;
    pthread_mutex_unlock(&solution_mutex);

  } while(_k<=MAX_STEPS && !found_solution);

  safe_queues::get_instance()->update_finished(safe_queues::FORWARD_CONDITION, true);

//  std::cout << "### Stopping FC" << std::endl;

  pthread_mutex_lock(&kill_mutex);
  pthread_cond_wait(&kill_cond, &kill_mutex);
  pthread_mutex_unlock(&kill_mutex);
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

inductive_step_thread::~inductive_step_thread()
{
//  std::cout << "### IS Dying" << std::endl;
}

void inductive_step_thread::run()
{
  bool res=0;
  bool found_solution=false;

  // We will do BMC for each k, until MAX_STEPS
  do
  {
    res=_bmc.run(_goto_functions);
    safe_queues::get_instance()->update_queue(safe_queues::INDUCTIVE_STEP, _k, res);

    _bmc.options.set_option("unwind", i2string(++_k));

    pthread_mutex_lock(&solution_mutex);
    found_solution=solution_found;
    pthread_mutex_unlock(&solution_mutex);

  } while(_k<=MAX_STEPS && !found_solution);

  safe_queues::get_instance()->update_finished(safe_queues::INDUCTIVE_STEP, true);

//  std::cout << "### Stopping IS" << std::endl;

  pthread_mutex_lock(&kill_mutex);
  pthread_cond_wait(&kill_cond, &kill_mutex);
  pthread_mutex_unlock(&kill_mutex);
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
//      std::cout << "### BASE_CASE" << std::endl;
      bc_queue[k]=res;
      break;

    case FORWARD_CONDITION:
//      std::cout << "### FORWARD CONDITION" << std::endl;
      fc_queue[k]=res;
      break;

    case INDUCTIVE_STEP:
//      std::cout << "### INDUCTIVE CONDITION" << std::endl;
      is_queue[k]=res;
      break;
  }

//  std::cout << "### BC _k " << k << " res " << bc_queue[k] << std::endl;
//  std::cout << "### IS _k " << k << " res " << fc_queue[k] << std::endl;
//  std::cout << "### FC _k " << k << " res " << is_queue[k] << std::endl;

  // Check if a solution was found
  if(bc_queue[k] != MAX_STEPS+1)
  {
    // If the base case is false
    if(bc_queue[k])
    {
//      std::cout << std::endl << "VERIFICATION FAILED BC" << std::endl;
      finalize_multithread();
    }

    if(fc_queue[k] != MAX_STEPS+1)
    {
      if(!fc_queue[k])
      {
//        std::cout << std::endl << "VERIFICATION SUCCESSFUL: BC + FC" << std::endl;
        finalize_multithread();
      }
    }

    if(is_queue[k] != MAX_STEPS+1)
    {
      if(!is_queue[k])
      {
//        std::cout << std::endl << "VERIFICATION SUCCESSFUL BC + IS" << std::endl;
        finalize_multithread();
      }
    }
  }

  pthread_mutex_unlock(&_mutex);
}

void safe_queues::update_finished(STEP s, bool finished)
{
  pthread_mutex_lock(&kill_mutex);

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
    pthread_mutex_lock(&main_mutex);
    pthread_cond_broadcast(&main_cond);
    pthread_mutex_unlock(&main_mutex);
  }

  pthread_mutex_unlock(&kill_mutex);
}
