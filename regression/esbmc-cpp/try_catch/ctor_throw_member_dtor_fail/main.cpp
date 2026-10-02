// [except.ctor]/3: the member constructed before the body throws is destroyed
// before the handler runs, which frees its buffer.
#include <cassert>
#include <cstdlib>

int dtors = 0;

struct Buf
{
  int *p;
  Buf() : p((int *)malloc(sizeof(int)))
  {
  }
  ~Buf()
  {
    free(p);
    ++dtors;
  }
};

struct W
{
  Buf b;
  W()
  {
    throw 1;
  }
};

int main()
{
  try
  {
    W w;
  }
  catch (int)
  {
  }
  assert(dtors == 0);
  return 0;
}
