#include <cassert>
#include <pthread.h>

// Companion to github_7768_race_store_after_throw: `g` keeps its old value in
// the handler, so this assertion fails.
int g = 5;

int thrower()
{
  throw 1;
}

void *t(void *)
{
  try
  {
    g = thrower();
  }
  catch (...)
  {
    assert(g == 6);
  }
  return nullptr;
}

int main()
{
  pthread_t a;
  pthread_create(&a, nullptr, t, nullptr);
  pthread_join(a, nullptr);
  return 0;
}
