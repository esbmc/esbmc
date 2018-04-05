#include <pthread.h>
#include <assert.h>
int a[3];
void* t1(void* arg) {
  a[1]=1;
  a[1]=2;
  assert(a[1]==2);
  return NULL;
}
void* t2(void* arg){
  a[0]=1;
  a[1]=3;
  a[2]=2;
  return NULL;
}
void* t3(void* arg){
  a[0]=4;
  a[2]=3;
  a[1]=3;
  return NULL;
}
int main(void){
  pthread_t id1, id2, id3;
  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
  pthread_create(&id3, NULL, t3, NULL);
  pthread_join(id1, NULL);
  pthread_join(id2, NULL);
  pthread_join(id3, NULL);
  return 0;
}
