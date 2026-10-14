#include <cassert>

// Catch a base class by value of a thrown derived object: the copy slices to
// the base subobject. (Single inheritance: base is at offset 0.)
struct Base
{
  int x;
  Base(int v) : x(v)
  {
  }
};
struct Der : Base
{
  int y;
  Der(int a, int b) : Base(a), y(b)
  {
  }
};

int main()
{
  try
  {
    throw Der(3, 4);
  }
  catch (Base b)
  {
    assert(b.x == 3);
    return 1;
  }
  return 0;
}
