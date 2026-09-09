// KNOWNBUG. Aggregate-initialising a class whose member type has a destructor
// runs one destructor too many. ESBMC materialises a temporary for the member,
// copies it into the object bitwise, and then destroys BOTH:
//
//     DECL M t;      FUNCTION_CALL: M(&t, 5)          <- construct t
//     DECL W a;
//     DECL M tmp$1;  FUNCTION_CALL: M(&tmp$1, &t)     <- copy into a helper
//                    ASSIGN a = { .m = tmp$1 }        <- copied into the object
//                    FUNCTION_CALL: ~M(&tmp$1)        <- helper destroyed
//                    FUNCTION_CALL: ~W(&a)            <- a.m destroyed too
//                    FUNCTION_CALL: ~M(&t)
//
// C++ constructs the member in place, so g++ runs two constructors and two
// destructors here; ESBMC runs two and three. The stored VALUE is correct --
// a.m.v == 5 verifies -- so only the lifetime accounting is wrong, which is
// why it shows up as a resource bug rather than a wrong result.
//
// It is not about temporaries: `M t(5); W a{t};` with a named argument is
// equally unbalanced (2 constructors, 3 destructors), as is `W a = W{M(5)}`.
// The counts below are for the temporary form, which is the smallest.
//
// This is the mechanism under regression/esbmc-cpp/cpp/shared_ptr_member_copy:
// the surplus ~M is a second shared_ptr __release(), so the refcount underflows
// and the control block is freed while an owner still holds it.
//
// The fix is to construct the member in place rather than through a helper --
// the same elision the existing array_init$ guard in convert_decl approximates
// for its own construction helper. aggregate_init_temp_controls pins the
// shapes that stay balanced.
#include <cassert>

int ctors = 0, dtors = 0;

struct M
{
  int v;
  M(int x) : v(x)
  {
    ++ctors;
  }
  M(const M &o) : v(o.v)
  {
    ++ctors;
  }
  ~M()
  {
    ++dtors;
  }
};

struct W
{
  M m;
};

int main()
{
  {
    W a{M(5)};
  }
  assert(ctors == dtors);
  return 0;
}
