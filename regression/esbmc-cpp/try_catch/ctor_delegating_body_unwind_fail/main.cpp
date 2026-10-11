#include <cassert>

int dtors = 0;

struct C
{
  C(bool)
  {
  }
  C(int body_throws) : C(false)
  {
    if (body_throws)
      throw 2;
  }
  ~C()
  {
    ++dtors;
  }
};

int main()
{
  try
  {
    C c(1);
  }
  catch (int)
  {
  }
  assert(dtors == 0);
  return 0;
}
