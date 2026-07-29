#include <cassert>

// Split out of gcc-template-tests/spec26 (GCC PR 23519, DR214).
//
// GCC resolves `b * a` to the free operator* and returns 6. Clang rejects the
// call as ambiguous, because under [temp.func.order] neither candidate is more
// specialised: the member template wins on the first parameter (B<A> vs T) and
// the free template wins on the second (A vs R). ESBMC uses Clang as its C++
// frontend, so it inherits that verdict and reports a parsing error.
//
// This test pins the behaviour: if the frontend ever silently accepts the call
// it would have to pick an arbitrary candidate, which is worse than rejecting.

struct A
{
    template<class T> int operator+(T&) { return 1;}
};

template<class T> struct B
{
  int operator-(A&) {return 2;}
  template<typename R> int operator*(R&) {return 3;}
};

template <typename T> int operator*(T &, A&){return 6;}

int main()
{
  A a;
  B<A> b;
  assert((b * a) == 6);
}
