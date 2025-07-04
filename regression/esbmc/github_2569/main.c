#include <assert.h>
extern int x;
int main(int argc, char **argv)
{
  if (nondet_int())
  {
    argc = 3;
  }
  else
  {
    argc = 2;
  }
  if (argc > 5)
  {
    x = 42;
  }
  assert(x == 42);
}
