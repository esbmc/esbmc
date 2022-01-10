#include <assert.h>

void *foo()
{
  assert(0);
  return 0;
}

void call_function(void *(*start_routine)(void *), void *arg)
{
  (*start_routine)(arg);
}

int main()
{
  call_function(&foo, 0);
  return 0;
}
