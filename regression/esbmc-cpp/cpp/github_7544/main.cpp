/* A heap array whose size expression references a data member is given a
 * symbolic array type; storing into it and casting the result reaches the
 * array-to-array arm of convert_typecast. Without that arm ESBMC aborts with
 * "Typecast for unexpected type" (esbmc/esbmc#7544, fixed by #7618 without a
 * test). The shape is load-bearing -- the loop, the conditional write through
 * an undefined extern, and the strcpy all contribute; simplifying any of them
 * can stop reaching the arm while the test still passes. */
#include <cassert>
#include <cstring>

extern bool nondet_bool();

class a
{
  int b;

public:
  char *g;
  a() : b(2), g(new char[b])
  {
    g[1] = '\0';
  }
  void operator()()
  {
    if (nondet_bool())
      g[0] = 'x';
    else
      g[0] = '\0';
  }
};

int main()
{
  a f;
  for (int i = 0; i < 2; i++)
  {
    f();
    char *dst = new char[2];
    strcpy(dst, f.g);
    /* Reads back through the converted cast: an arm that dropped the source
     * would still avoid the abort, but would fail here. */
    assert(dst[0] == f.g[0]);
  }
}
