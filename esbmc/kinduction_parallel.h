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

const unsigned int MAX_STEPS=50;

class base_case
{
  public:
    base_case(bmct &bmc,
        goto_functionst &goto_functions);

    void startSolving();

  private:
    unsigned int _k;
    bmct &_bmc;
    goto_functionst &_goto_functions;
};

class forward_condition
{
  public:
    forward_condition(bmct &bmc,
        goto_functionst &goto_functions);

    void startSolving();

  private:
    unsigned int _k;
    bmct &_bmc;
    goto_functionst &_goto_functions;
};

class inductive_step
{
  public:
    inductive_step(bmct &bmc,
        goto_functionst &goto_functions);

    void startSolving();

  private:
    unsigned int _k;
    bmct &_bmc;
    goto_functionst &_goto_functions;
};

class safe_queues
{
  public:
    static safe_queues *get_instance();

    enum STEP { BASE_CASE, FORWARD_CONDITION, INDUCTIVE_STEP };

    void update_queue(STEP s, unsigned int k, int res);
    void update_finished(STEP s, bool finished);

  private:
    safe_queues();
    ~safe_queues();

    short bc_queue[MAX_STEPS];
    short fc_queue[MAX_STEPS];
    short is_queue[MAX_STEPS];

    pthread_mutex_t _mutex;

    static safe_queues* instance;

    bool bc_finished, fc_finished, is_finished;

    safe_queues(safe_queues const &);
    void operator=(safe_queues const &);
};
#endif /* K_INDUCTION_PARALLEL_H_ */
