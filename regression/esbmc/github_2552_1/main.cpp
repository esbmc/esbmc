#include <assert.h>
extern void __VERIFIER_assume(int cond);
extern int x;
int main(int argc, char **argv)
{
  __VERIFIER_assume(argc == 6);
  if (argc > 5)
    x = 42;
  assert(x == 42);
}
