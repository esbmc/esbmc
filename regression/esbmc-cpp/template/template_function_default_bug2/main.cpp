#include <assert.h>
//#include <iostream>

template <typename S, typename T> void f(S s = 0, T t = 0)
{
  assert(s==2);
  assert(t==0);
}

int main() {
  f<int,double>(2);
  f<int,double>(2,3);
  return 0;
}
