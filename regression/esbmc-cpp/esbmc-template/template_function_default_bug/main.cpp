#include <assert.h>
//#include <iostream>

template <typename S, typename T> void f(S s = 0, T t = 0)
{
  assert(s==0);
  assert(t==1);
}

int main() {
  f<int,char>();     // f<int,char>(0,0)

}
