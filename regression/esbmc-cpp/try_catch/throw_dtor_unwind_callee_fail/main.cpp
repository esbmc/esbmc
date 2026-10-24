// An exception leaving a callee destroys the caller's automatic objects
// ([except.ctor]/1), so freeing the buffer again in the handler is a double
// free.
#include <cstdlib>

struct Buffer
{
  char *p;
  Buffer() : p((char *)malloc(4))
  {
  }
  ~Buffer()
  {
    free(p);
  }
};

char *seen;

void thrower()
{
  throw 1;
}

void fill()
{
  Buffer b;
  seen = b.p;
  thrower();
}

int main()
{
  try
  {
    fill();
  }
  catch (int)
  {
    free(seen);
  }
  return 0;
}
