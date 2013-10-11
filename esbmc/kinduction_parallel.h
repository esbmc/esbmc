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

extern pthread_mutex_t main_mutex;
extern pthread_cond_t main_cond;

class base_case_thread : public Thread
{
  public:
    base_case_thread(bmct &bmc,
        goto_functionst &goto_functions);

  protected:
    virtual void run();

  private:
    unsigned int _k;
    bmct &_bmc;
    goto_functionst &_goto_functions;
};

class forward_condition_thread : public Thread
{
  public:
    forward_condition_thread(bmct &bmc,
        goto_functionst &goto_functions);

  protected:
    virtual void run();

  private:
    unsigned int _k;
    bmct &_bmc;
    goto_functionst &_goto_functions;
};

class inductive_step_thread : public Thread
{
  public:
    inductive_step_thread(bmct &bmc,
        goto_functionst &goto_functions);

  protected:
    virtual void run();

  private:
    unsigned int _k;
    bmct &_bmc;
    goto_functionst &_goto_functions;
};

class safe_queues
{
  public:
    static safe_queues *get_instance();

    void update_bc_queue(unsigned int k, int res);
    void update_fc_queue(unsigned int k, int res);
    void update_is_queue(unsigned int k, int res);

  private:
    safe_queues();
    ~safe_queues();

    short bc_queue[50];
    short fc_queue[50];
    short is_queue[50];

    pthread_mutex_t _bcMutex;
    pthread_mutex_t _fcMutex;
    pthread_mutex_t _isMutex;

    static safe_queues* instance;

    safe_queues(safe_queues const &);
    void operator=(safe_queues const &);
};
#endif /* K_INDUCTION_PARALLEL_H_ */
