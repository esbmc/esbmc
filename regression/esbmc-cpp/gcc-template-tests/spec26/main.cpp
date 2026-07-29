#include <cassert>
// dg-do run
// Copyright (C) 2005 Free Software Foundation, Inc.
// Contributed by Nathan Sidwell 16 Sep 2005 <nathan@codesourcery.com>

// PR 23519  template specialization ordering (DR214)
// Origin:  Maxim Yegorushkin <maxim.yegorushkin@gmail.com>

// The original GCC test also checked `b * a == 6`. Clang — and hence the ESBMC
// frontend — rejects that call as ambiguous: neither the member template
// B<T>::operator*(R&) nor the free operator*(T&, A&) is more specialised than
// the other under [temp.func.order]. That divergence is pinned separately in
// spec26-dr214-ambiguous/.

struct A
{
    template<class T> int operator+(T&) { return 1;}
};

template<class T> struct B
{
  int operator-(A&) {return 2;}
  template<typename R> int operator*(R&) {return 3;}
};

template <typename T, typename R> int operator-(B<T>, R&) {return 4;}
template<class T> int operator+(A&, B<T>&) {return 5;}

int main()
{
  A a;
  B<A> b;
  if ((a + b) != 5)
    assert(0 == (1));

  if ((b - a) != 2)
    assert(0 == (2));
}
