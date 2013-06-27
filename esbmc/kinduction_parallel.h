/*
 * K_induction_parallel.h
 *
 *  Created on: 27/06/2013
 *      Author: ceteli04
 */

#ifndef K_INDUCTION_PARALLEL_H_
#define K_INDUCTION_PARALLEL_H_

#include "thread.h"

// This will be the base class for k induction threads
class kinduction_thread : public Thread
{
  public:
    kinduction_thread();

  protected:
    unsigned int k;
};

class base_case_thread : public kinduction_thread
{
  public:
    base_case_thread();
    virtual void run();
};

class forward_condition_thread : public kinduction_thread
{
  public:
    forward_condition_thread();
    virtual void run();
};

class inductive_step_thread : public kinduction_thread
{
  public:
    inductive_step_thread();
    virtual void run();
};

#endif /* K_INDUCTION_PARALLEL_H_ */
