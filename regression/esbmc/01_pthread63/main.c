#include <pthread.h>
#define N 2

int a[N], i, j;

void *t1(void *arg)
{
  i = 0;
  a[i] = 2;
}

void *t2(void *arg)
{
  j = 1;
  a[j]=3;
}

//void *t3(void *arg)
//{
//  k = 2;
//  a[k]=5;
//}

int main()
{
  pthread_t id1, id2; //, id3;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
 // pthread_create(&id3, NULL, t3, NULL);

  pthread_join(id1, NULL);
  pthread_join(id2, NULL);
  //pthread_join(id3, NULL);

  assert(a[i]==2 && a[j]==3 /*&& a[k]==5*/);

}
