#include <cassert>

int calls = 0;
int dtor_runs = 0;

struct Guard
{
  ~Guard()
  {
    ++dtor_runs;
  }
};

bool probe(int x)
{
  ++calls;
  return x > 0;
}

// A side-effecting guard: the dispatcher delegates the whole if-statement to
// convert_ifthenelse while the statements around it convert natively. The
// guard must be evaluated exactly once, before the branch, and the enclosing
// scope's destructor must still run exactly once.
int classify(int x)
{
  Guard g;
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
  assert(calls == 1);

  assert(classify(-5) == 2);
  assert(calls == 2);

  assert(dtor_runs == 2);
  return 0;
}
