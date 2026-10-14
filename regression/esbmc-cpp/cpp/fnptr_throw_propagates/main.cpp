#include <cassert>

struct E
{
};

static void boom()
{
  throw E();
}
static void safe()
{
}

int main()
{
  int nd;
  void (*f)() = nd ? &boom : &safe;

  bool caught = false;
  try
  {
    f();
  }
  catch (E &)
  {
    caught = true;
  }

  // boom() is address-taken and throws, so the call through f must still
  // propagate: the may-throw analysis may not drop an indirect call whose
  // possible targets include a throwing function.
  assert(!caught);
  return 0;
}
