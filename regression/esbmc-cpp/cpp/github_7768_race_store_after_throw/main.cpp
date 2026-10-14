#include <cassert>
#include <pthread.h>

// A callee that throws does not assign the call's result, so `g` must keep
// its value in the handler also when race checks are instrumented.
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
    assert(g == 5);
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
