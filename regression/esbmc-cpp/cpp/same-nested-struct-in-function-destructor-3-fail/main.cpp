
#include <cassert>

int g = 0;
int h = 0;
int i = 0;
struct foo
{
  foo()
  {
    struct axy
    {
      ~axy()
      {
        g = 2;
      }
    } __guard;
  }
};
struct bar
{
  bar()
  {
    struct axy
    {
      ~axy()
      {
        h = 2;
      }
    } __guard;
  }
};
struct foobar
{
  foobar()
  {
    struct axy
    {
      ~axy()
      {
        i = 2;
      }
    } __guard;
  }
};

int main()
{
  {
    foo f;
    bar b;
    foobar fb;
  }
  assert(g == 22); // should be 2
  assert(h == 22); // should be 2
  assert(i == 22); // should be 2
}
