#include <cassert>

struct A
{
  int value;
  explicit A(int v) : value(v)
  {
  }
};

struct B
{
  int value;
  explicit B(int v) : value(v)
  {
  }
};

int main()
{
  try
  {
    throw A(1);
  }
  catch (A &)
  {
    try
    {
      throw B(2);
    }
    catch (B &)
    {
    }

    try
    {
      throw;
    }
    catch (A &a)
    {
      assert(a.value == 1);
      return 0;
    }
    catch (...)
    {
      assert(false);
    }
  }

  assert(false);
}
