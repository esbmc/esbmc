#include <cassert>
// Companion to gcc-template-tests/spec26, which is rejected as a whole because
// its `b * a` is ambiguous. These are the DR214 partial-ordering checks from
// that test that are well-formed, so the ordering behaviour stays covered.

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
  // The free operator+ is more specialised than A's member template.
  if ((a + b) != 5)
    assert(0 == (1));

  // B's non-template member beats the free operator- template.
  if ((b - a) != 2)
    assert(0 == (2));
}
