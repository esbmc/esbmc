// KNOWNBUG. The user-visible cost of the skipped array-element destructors:
// each element frees its allocation in ~R, but the destructors do not run, so
// --memory-leak-check reports a leak per element on a program that leaks
// nothing. g++ runs this cleanly under valgrind.
#include <cassert>

struct R
{
  int *p;
  R() : p(new int(1))
  {
  }
  ~R()
  {
    delete p;
  }
};

int main()
{
  {
    R a[2];
    assert(*a[0].p == 1);
  }
  return 0;
}
