// Author: Chao Wang

// Date:  7/22/2009

// Description:  Test for atomicity violations

// This pattern comes from a real bug in the Mozilla Application
// Suite. When thread 2 violates the atomicity of thread 1's accesses
// to getCurrentScript, the program crashes.

// Note: modified from Figure 1 of the AVIO paper.  


#include <stdio.h>
#include <pthread.h>

#define InitLock(A) pthread_mutex_init  (&A,NULL)
#define Lock(A)     pthread_mutex_lock  (&A)
#define UnLock(A)   pthread_mutex_unlock(&A)


typedef struct _nsSpt { void(*compile)(); } nsSpt;


pthread_mutex_t  l;
nsSpt            global_script;
nsSpt*           gCurrentScript = NULL;

//-----------------------------------
#define NUM_DATA  1
pthread_mutex_t  data_l;
int data[NUM_DATA];

void data_init()
{
  int i;
  for (i = 0; i< NUM_DATA; i++) {
    Lock(data_l);
    data[i] = 0;
    UnLock(data_l);
  }
}

void data_access_1() 
{
  int i;
  int a;
  for (i = 0; i< NUM_DATA; i++) {
    pthread_mutex_lock(&data_l);
    a = data[i];
    data[i] = i;
    pthread_mutex_unlock(&data_l);
  }
}
  
void data_access_2() 
{
  int i;
  int a;
  for (i = 0; i< NUM_DATA; i++) {
    Lock(data_l);
    a = data[i];
    data[i] = i * 10;
    UnLock(data_l);
  }
}
//-----------------------------------


//-----------------------------------
// used for specifying atomic regions  
extern void inspect_atom_start(int);
extern void inspect_atom_end(int);
//-----------------------------------


void Dummy()
{
//  printf("executing Dummy() !\n");
}

void LaunchLoad(nsSpt* aspt)
{
//  printf("executing LaunchLoad() !\n");
}

void LoadScript(nsSpt* aspt) 
{
//  printf("executing LoadScript() begin ...\n");

  Lock(l);

  gCurrentScript = aspt;

  LaunchLoad(aspt);

  UnLock(l);

//  printf("executing LoadScript()   end \n");
}

void OnLoadComplete() 
{
  nsSpt* spt;
  // call back function of LaunchLoad */

//  printf("executing OnLoadComplete() begin ...\n");

  Lock(l);

  spt = gCurrentScript;

  if (spt != 0) {
    gCurrentScript->compile();
  }else {
    //assert(0);
  }

  UnLock(l);

//  printf("executing OnLoadComplete()  end \n");
}

void* thread_routine1(void* aspt)
{
//  printf("executing thread_routine1() begin ... \n");

//  inspect_atom_start(0);

  data_access_1();

  LoadScript(aspt);

  data_access_1();

  OnLoadComplete();

  data_access_1();

//  inspect_atom_end(0);

//  printf("executing thread_routine1()   end \n\n");
}

void* thread_routine2(void* arg)
{
  sleep(1);
//  printf("executing thread_routine2() begin ... \n");

  Lock(l);

  data_access_2();
  
  gCurrentScript = NULL;

  data_access_2();

  UnLock(l);

//  printf("executing thread_routine2()   end \n\n");
}


int main()
{
  pthread_t  t1, t2;

  InitLock(l);
  InitLock(data_l);

  data_init();

  global_script.compile = Dummy;

  pthread_create(&t1, 0, thread_routine1, &global_script);
  pthread_create(&t2, 0, thread_routine2, 0);
  
  pthread_join(t1, 0);
  pthread_join(t2, 0);

  return 0;
}
