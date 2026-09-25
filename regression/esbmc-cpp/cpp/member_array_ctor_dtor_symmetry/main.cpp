// Every element of a class-typed member array must be constructed before it is
// destroyed. clang_cpp_convert.cpp bounds per-element construction at 8
// elements but leaves destruction unbounded, so elements 8.. are destroyed
// without ever having been constructed.
#include <cassert>

int marker;

struct E
{
  int *p;
  E()
  {
    p = &marker;
  }
  ~E()
  {
    assert(p == &marker);
  }
};

struct H
{
  E buf[20];
};

int main()
{
  H h;
  return 0;
}
