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

const unsigned int MAX_STEPS=50;

class base_caset
{
  public:
    base_caset(bmct &bmc,
        goto_functionst &goto_functions);

    bool startSolving();

  private:
    unsigned int _k;
    bmct &_bmc;
    goto_functionst &_goto_functions;
};

class forward_conditiont
{
  public:
    forward_conditiont(bmct &bmc,
        goto_functionst &goto_functions);

    bool startSolving();

  private:
    unsigned int _k;
    bmct &_bmc;
    goto_functionst &_goto_functions;
};

class inductive_stept
{
  public:
    inductive_stept(bmct &bmc,
        goto_functionst &goto_functions);

    bool startSolving();

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
