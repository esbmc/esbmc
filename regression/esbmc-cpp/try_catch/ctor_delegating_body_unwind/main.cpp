#include <cassert>

int dtors = 0;

struct C
{
  C(bool target_throws)
  {
    if (target_throws)
      throw 1;
  }
  C(int body_throws) : C(false)
  {
    if (body_throws)
      throw 2;
  }
  C(char target_throws) : C(target_throws != 0)
  {
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
  assert(dtors == 1);

  try
  {
    C c('\1');
  }
  catch (int)
  {
  }
  assert(dtors == 1);

  {
    C c(0);
  }
  assert(dtors == 2);
  return 0;
}
