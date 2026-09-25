#include <cassert>

int calls = 0;

bool probe(int x)
{
  ++calls;
  return x > 0;
}

int classify(int x)
{
  int r = 0;
  if (probe(x))
    r = 1;
  else
    r = 2;
  return r;
}

int main()
{
  assert(classify(5) == 1);
  // The guard is evaluated exactly once, so this must fail rather than be
  // vacuously true: it pins that the delegated if does not double-evaluate
  // or drop the guard's side effect.
  assert(calls == 2);
  return 0;
}
