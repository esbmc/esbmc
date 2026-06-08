// Race-safe companion to github_4715_rwset_fail: the same three IREP2
// rw_set paths (array index / member / if-guarded) are exercised, but
// each thread writes its OWN disjoint locations. The only shared
// global is `cond`, which is read-only across the two threads
// (concurrent reads are not races). --data-races-check must NOT fire.

#include <pthread.h>

int arr_t1[2];
int arr_t2[2];

struct S
{
  int x;
  int y;
};
struct S s_t1;
struct S s_t2;

int cond; // shared but read-only across the two threads

void *thread1(void *_)
{
  arr_t1[0] = 1;
  s_t1.x = 1;
  if (cond)
    s_t1.y = 1;
  return 0;
}

void *thread2(void *_)
{
  arr_t2[0] = 2;
  s_t2.x = 2;
  if (cond)
    s_t2.y = 2;
  return 0;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, 0, thread1, 0);
  pthread_create(&t2, 0, thread2, 0);
  return 0;
}
