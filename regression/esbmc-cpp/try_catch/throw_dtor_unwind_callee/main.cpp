// An exception leaving a callee destroys the caller's automatic objects too,
// newest first ([except.ctor]/1). An object whose constructor throws was
// never constructed, so it is not destroyed.
#include <cassert>

int order[4];
int n = 0;

struct Guard
{
  int id;
  Guard(int i) : id(i)
  {
  }
  ~Guard()
  {
    order[n++] = id;
  }
};

struct Throws
{
  Throws()
  {
    throw 1;
  }
  ~Throws()
  {
    order[n++] = 99;
  }
};

void thrower()
{
  throw 1;
}

void middle()
{
  Guard g(3);
  thrower();
}

void outer()
{
  Guard a(1);
  Guard b(2);
  middle();
}

void consume(const Throws &)
{
}

void temporary()
{
  Guard c(4);
  consume(Throws());
}

int main()
{
  try
  {
    outer();
  }
  catch (int)
  {
  }
  assert(n == 3 && order[0] == 3 && order[1] == 2 && order[2] == 1);

  n = 0;
  try
  {
    temporary();
  }
  catch (int)
  {
  }
  assert(n == 1 && order[0] == 4);
  return 0;
}
