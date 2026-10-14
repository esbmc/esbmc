// esbmc/esbmc#4715: a thunk's argument symbols are named from the code type's
// argument base names, which add_thunk_method_arguments reads off the IREP2
// type. If those stop being carried, every argument collapses onto one symbol.
#include <cassert>

struct Base
{
  virtual ~Base()
  {
  }
  virtual int f(int, int)
  {
    return 0;
  }
};

struct Derived : Base
{
  int f(int a, int b) override
  {
    return a - b;
  }
};

int main()
{
  Derived d;
  Base *p = &d;
  assert(p->f(5, 2) == 3);
  return 0;
}
