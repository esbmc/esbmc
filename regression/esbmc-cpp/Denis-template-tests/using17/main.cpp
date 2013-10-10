#include <cassert>
// PR c++/14258
// { dg-do run }

template<typename T>
struct A 
{
  typedef T type;
  typedef A type2;
};
                                                                               
template<typename T>
struct B : A<T> 
{
  using typename A<T>::type;
  type t;

  using typename A<T>::type2;

  type f()
  {
    type i = 1;
    return i;
  }
};

int main()
{
  B<int>::type t = 4;
  if (t != 4)
    assert(0);

  B<double> b;
  b.t = 3;
  if (b.t != 3)
    assert(0);

  B<long> b2;
  if (b2.f() != 1)
    assert(0);

  B<double>::type2::type tt = 12;
  if (tt != 12)
    assert(0);
}

