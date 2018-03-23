#ifndef _ECSC_H
#define _ECSC_H

int global_time=0; 			//global timer
int nondet_int(); 			//returns a non-deterministic interger value
void __ESBMC_atomic_begin();//begin block of the atomic section
void __ESBMC_atomic_end();	//end block of the atomic section
void __ESBMC_yield();		//force esbmc's scheduler to make a context switch

/**
*   @brief Procedure used to wait for an event
*
*   @retval void.
*/

void wait_event(_Bool e)
{
__ESBMC_atomic_begin();
  if (!e)
    __ESBMC_assume(0);
__ESBMC_atomic_end();
}

/**
*   @brief Procedure used to wait for a specific time
*
*   @retval void.
*/

void wait_time(int t)
{
  __ESBMC_atomic_begin();
  static int local_time = 0;
  local_time = global_time + t;
  __ESBMC_atomic_end();
  __ESBMC_yield();
  __ESBMC_atomic_begin();
  if (!(global_time==local_time))
    __ESBMC_assume(0);  
  __ESBMC_atomic_end();
}

/**
*   @brief Function used to notify an event
*
*   @retval This function returns 1 to notify the event.
*/

int notify_event(void)
{
__ESBMC_atomic_begin();
  return 1;
__ESBMC_atomic_end();
}

#endif /* _ECSC_H */

