#include <pthread.h>
#include <assert.h>

int flag[2], turn, x, i;
int nondet_int();

void *t1(void *arg) {
  flag[0] = 1;
  turn = 1;
  while (flag[1] == 1 && turn == 1) {};
  //critical section
  if (i==1) x=1;
  //end of critical section
  flag[0] = 0;  

  return NULL;
}

void *t2(void *arg) {
  flag[1] = 1;
  turn = 0;
  while (flag[0] == 1 && turn == 0) {};
  //critical section
  if (i==2) x=3;
  //end of critical section
  flag[1] = 0;  

  return NULL;
}

int main(void) {

  pthread_t id1, id2;
  i = nondet_int();
  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);

  if (flag[0]==0 && flag[1]==0)
    assert(x==0 || x==1 || x==2); //this should fail

  return 0;
}
