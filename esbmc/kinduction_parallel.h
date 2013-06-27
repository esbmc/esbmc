/*
 * K_induction_parallel.h
 *
 *  Created on: 27/06/2013
 *      Author: ceteli04
 */

#ifndef K_INDUCTION_PARALLEL_H_
#define K_INDUCTION_PARALLEL_H_

#include "thread.h"

class kinduction_thread : public Thread
{
  public:
    kinduction_thread();

    virtual void run();
};

#endif /* K_INDUCTION_PARALLEL_H_ */
