#include <pthread.h>
#include <assert.h>

int x=0, k=0;
int join1=0, join2=0;
void   __ESBMC_yield();
void check_property(void) {
  if (join1==1 && join2==1)
    assert(x==0); //assert(x==1) //10 failed of 45
				  //assert(x==0) //35 failed of 45
}

void* t1(void* arg) {
  x++;
  if (x>1)
    x--;
  join1=1;
//  check_property();
  return NULL;
}

void* t2(void* arg) {
  x++;
  if (x>1)
  {
    x--;
  }
  join2=1;
//  check_property();
  return NULL;
}

int main(void) {
  pthread_t id1, id2;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);

//  check_property();
  if (join1==1 && join2==1)
    assert(x==1); //assert(x==1) //10 failed of 45
				  //assert(x==0) //35 failed of 45

  return 0;
}
