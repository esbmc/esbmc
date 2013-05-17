#include <cassert>

class X 
{
  public:
    X(int x): i(x){}
    int i;
};

template < X x = X(1), class T = X() >
class A
{
public:
  A():v(x){}
  T v;
};

int main()
{
  A<> a;
  assert(a.v.i==1);
  return 0;
}

