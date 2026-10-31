// esbmc/esbmc#9376: a null C* converted to its non-first base B* stays null,
// so the assertion fails.
#include <cassert>

struct A
{
  int a = 1;
};
struct B
{
  int b = 2;
};
struct C : A, B
{
};

int main()
{
  C *pc = nullptr;
  B *pb = pc;
  assert(pb != nullptr);
  return 0;
}
