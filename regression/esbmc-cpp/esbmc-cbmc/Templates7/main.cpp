#include<cassert>

template <class T>
class A
{
public:
  T t;
};

int main()
{
  A<A<double> > a1;
  A<A<int> > a2;

  a1.t.t=3.0;
  assert(a1.t.t==3.0);

  a2.t.t=2;
  assert(a2.t.t==2);
}
