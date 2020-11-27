#include <stdio.h>
#include <pthread.h>
#include <assert.h>

int token;
_Bool event_E1=0, event_E2=0, event_E3=0, event_E4=0, event_E5=0, event_EM=0;
int nondet_int();
void __ESBMC_atomic_begin();
void __ESBMC_atomic_end();

void wait_event(_Bool e1, _Bool *e2)
{
__ESBMC_atomic_begin();
  _Bool unlocked = (e1==0);
  if (unlocked)
    __ESBMC_assume(0);
  else
    e2=0;
__ESBMC_atomic_end();
}

int notify_event(void)
{
__ESBMC_atomic_begin();
  return 1;
__ESBMC_atomic_end();
}

void* SC_THREAD_master(void* arg)
{
  int local;

//  while(1) { 
    token = nondet_int(); 
    local = token;
    event_E1 = notify_event(); 
    wait_event(event_EM, &event_EM); 
    assert(token == (local + 3));
//  }

//  return NULL;
}

void* SC_THREAD_transmit1(void* arg)
{
//  while(1) { 
    wait_event(event_E1, &event_E1); 
    token = token + 1;
    event_E2 = notify_event(); 
//  }
//  return NULL;
}

void* SC_THREAD_transmit2(void* arg)
{
//  while(1) { 
    wait_event(event_E2, &event_E2); 
    token = token + 1;
    event_E3 = notify_event(); 
//  }
//  return NULL;
}

void* SC_THREAD_transmit3(void* arg)
{
//  while(1) { 
    wait_event(event_E3, &event_E3); 
    token = token + 1;
    event_EM = notify_event(); 
//  }
//  return NULL;
}
#if 0
void* SC_THREAD_transmit4(void* arg)
{
//  while(1) { 
    wait_event(event_E4, &event_E4); 
    token = token + 1;
    event_EM = notify_event(); 
//  }
//  return NULL;
}

void* SC_THREAD_transmit5(void* arg)
{
//  while(1) { 
    wait_event(event_E5,event_E5); 
    token = token + 1;
    event_EM = notify_event(); 
//  }
//  return NULL;
}
#endif
int main(void)
{
  pthread_t id[6];

  pthread_create(&id[0], NULL, SC_THREAD_master, NULL);
  pthread_create(&id[1], NULL, SC_THREAD_transmit1, NULL);
  pthread_create(&id[2], NULL, SC_THREAD_transmit2, NULL);
  pthread_create(&id[3], NULL, SC_THREAD_transmit3, NULL);
//  pthread_create(&id[4], NULL, SC_THREAD_transmit4, NULL);
//  pthread_create(&id[5], NULL, SC_THREAD_transmit5, NULL);
#if 0
  pthread_join(id[0], NULL);
  pthread_join(id[1], NULL);
  pthread_join(id[2], NULL);
  pthread_join(id[3], NULL);
  pthread_join(id[4], NULL);
  pthread_join(id[5], NULL);
#endif
  return 0;
}
