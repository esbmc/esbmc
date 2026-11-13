#include <assert.h>
#include <pthread.h>

// The bug needs about ten context switches: any default context bound in
// --sv-comp's proof pass would miss it and report SUCCESSFUL (#8220).
void reach_error(void) { assert(0); }

int s = 0;

void *t1(void *arg)
{
  if (s == 0) s = 1;
  if (s == 2) s = 3;
  if (s == 4) s = 5;
  if (s == 6) s = 7;
  if (s == 8) s = 9;
  return 0;
}

void *t2(void *arg)
{
  if (s == 1) s = 2;
  if (s == 3) s = 4;
  if (s == 5) s = 6;
  if (s == 7) s = 8;
  if (s == 9) reach_error();
  return 0;
}

int main(void)
{
  pthread_t a, b;
  pthread_create(&a, 0, t1, 0);
  pthread_create(&b, 0, t2, 0);
  pthread_join(a, 0);
  pthread_join(b, 0);
  return 0;
}
