#include <cassert>
// PR c++/20208
// { dg-do run }
// { dg-options "-O2" }

extern "C" void abort();

template <typename T>
inline void *Foo (T arg) {  &arg[0]; }

int main () {
  int bry[2];
  if (Foo<int[2]>(bry) != bry)
    assert(0);
}
