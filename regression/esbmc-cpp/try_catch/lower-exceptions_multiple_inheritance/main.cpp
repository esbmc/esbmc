#include <cassert>

// The frontend emits a flattened throw list [D, A, B] for `struct D : A, B`.
// Registering it must record D<:A and D<:B but NOT A<:B, otherwise a later
// `throw A()` could wrongly match `catch (B&)`.
struct A
{
  int a;
};
struct B
{
  int b;
};
struct D : A, B
{
  int d;
};

void registers_chain()
{
  throw D(); // its [D, A, B] throw list teaches the type-id registry
}

int main()
{
  try
  {
    throw A();
  }
  catch (B &) // A is not a B
  {
    assert(false);
  }
  catch (A &)
  {
    return 1;
  }
  return 0;
}
