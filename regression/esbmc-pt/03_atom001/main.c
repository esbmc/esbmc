// Author: Chao Wang

// Date:  7/22/2009

// Description:  Test for atomicity violations

// This pattern comes from a real bug in the Mozilla Application
// Suite. When thread 2 violates the atomicity of thread 1's accesses
// to getCurrentScript, the program crashes.

// Note: modified from Figure 1 of the AVIO paper.  


#include <stdio.h>
#include <pthread.h>

extern void inspect_atom_start(int);
extern void inspect_atom_end(int);

void __ESBMC_atomic_begin();
void __ESBMC_atomic_end();

#define InitLock(A) pthread_mutex_init  (&A,0)
#define Lock(A)     pthread_mutex_lock  (&A)
#define UnLock(A)   pthread_mutex_unlock(&A)

typedef struct _nsSpt { void(*compile)(); } nsSpt;


pthread_mutex_t  l;
nsSpt            global_script;
nsSpt*           gCurrentScript = NULL;

void Dummy()
{
  //printf("executing Dummy() !\n");
}

void LaunchLoad(nsSpt* aspt)
{
  //printf("executing LaunchLoad() !\n");
}

void LoadScript(nsSpt* aspt) 
{
  //printf("executing LoadScript() begin ...\n");

  Lock(l);

  gCurrentScript = aspt;

  LaunchLoad(aspt);

  UnLock(l);

  //printf("executing LoadScript()   end \n");
}

void OnLoadComplete() 
{
  nsSpt* spt;
  // call back function of LaunchLoad */

  //printf("executing OnLoadComplete() begin ...\n");

  Lock(l);

  spt = gCurrentScript;

  if (spt != 0) {
    gCurrentScript->compile();
  }else {
    assert(0);
  }

  UnLock(l);

  //printf("executing OnLoadComplete()  end \n");
}

void* thread_routine1(void* aspt)
{
  //printf("executing thread_routine1() begin ... \n");

  {
    //inspect_atom_start(1);
    
    LoadScript(aspt);
    
    OnLoadComplete();
    

    //inspect_atom_end(1);
  }

  //printf("executing thread_routine1()   end \n\n");
}

void* thread_routine2(void* arg)
{
  sleep(1);
  //printf("executing thread_routine2() begin ... \n");

  Lock(l);
  
  gCurrentScript = NULL;

  UnLock(l);

  //printf("executing thread_routine2()   end \n\n");
}


int main()
{
  pthread_t  t1, t2;

  InitLock(l);

  global_script.compile = Dummy;

  pthread_create(&t1, 0, thread_routine1, &global_script);
  pthread_create(&t2, 0, thread_routine2, 0);
  
  pthread_join(t1, 0);
  pthread_join(t2, 0);

  return 0;
}
