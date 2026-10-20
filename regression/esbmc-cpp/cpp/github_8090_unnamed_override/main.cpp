// github #8090: an override declared with unnamed parameters made every thunk
// argument share one symbol, so the thunk forwarded the last actual for all.
#include <cassert>

struct Base
{
  virtual ~Base() {}
  virtual int f(int, int, int) { return 0; }
};

struct Derived : Base
{
  int f(int, int, int c) override;
};

int Derived::f(int a, int b, int c) { return a - b * c; }

int main()
{
  Derived d;
  Base *b = &d;
  assert(b->f(7, 2, 3) == 1);
}
