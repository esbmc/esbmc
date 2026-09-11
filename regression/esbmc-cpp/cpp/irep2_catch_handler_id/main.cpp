// The handler's exception id is computed at conversion time, so it survives the
// migrate seam: code_block2t has no type to carry the catch type on
// (docs/roadmap/scope-clang-cpp-irep2.md §3.13). Computed later, the IREP2
// adjust pass sees an empty type and remove_exceptions rejects the handler.
#include <cassert>

struct E
{
  int v;
  E() : v(7)
  {
  }
};

int main()
{
  int caught = 0;

  try
  {
    throw E();
  }
  catch (const E &e)
  {
    caught = e.v;
  }

  int any = 0;
  try
  {
    throw 1;
  }
  catch (...)
  {
    any = 1;
  }

  assert(caught == 7 && any == 1);
}
