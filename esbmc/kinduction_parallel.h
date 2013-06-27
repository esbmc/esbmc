/*
 * K_induction_parallel.h
 *
 *  Created on: 27/06/2013
 *      Author: ceteli04
 */

#ifndef K_INDUCTION_PARALLEL_H_
#define K_INDUCTION_PARALLEL_H_

#include "thread.h"

#include <goto-programs/goto_functions.h>
#include "bmc.h"

// This will be the base class for k induction threads
class kinduction_thread : public Thread
{
  public:
    kinduction_thread(bmct &bmc, goto_functionst &goto_functions);

  protected:
    unsigned int k;
    bmct &_bmc;
    goto_functionst &_goto_functions;
};

class base_case_thread : public kinduction_thread
{
  public:
    base_case_thread(bmct &bmc, goto_functionst &goto_functions);
    virtual void run();
};

class forward_condition_thread : public kinduction_thread
{
  public:
    forward_condition_thread(bmct &bmc, goto_functionst &goto_functions);
    virtual void run();
};

class inductive_step_thread : public kinduction_thread
{
  public:
    inductive_step_thread(bmct &bmc, goto_functionst &goto_functions);
    virtual void run();
};

class safe_queues
{
  public:
    static safe_queues *get_instance();

    pthread_mutex_t _bcMutex;
    pthread_mutex_t _fcMutex;
    pthread_mutex_t _isMutex;

  private:
    safe_queues();
    ~safe_queues();

    short bc_queue[50];
    short fc_queue[50];
    short is_queue[50];

    static safe_queues* instance;

    safe_queues(safe_queues const &);
    void operator=(safe_queues const &);
};
#endif /* K_INDUCTION_PARALLEL_H_ */
