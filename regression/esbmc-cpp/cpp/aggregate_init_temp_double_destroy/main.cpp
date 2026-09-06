// KNOWNBUG. Aggregate-initialising a class from a temporary of a member type
// that has a destructor destroys that member TWICE: ESBMC runs 1 constructor
// and 2 destructors here, where g++ runs 1 and 1 ([class.temporary] and the
// C++17 guaranteed elision of a prvalue initialiser mean one object exists).
//
// The elided temporary keeps its destructor call in addition to the member's,
// so the count is unbalanced rather than merely imprecise. It is a genuine
// double-destroy, which is why it corrupts anything doing lifetime accounting:
// for std::shared_ptr the extra call is a second __release(), the refcount
// underflows, the control block is deleted while an owner still holds it, and
// the read is reported as an "invalidated dynamic object"
// (regression/esbmc-cpp/cpp/shared_ptr_member_copy).
//
// Neither templates nor a copy are involved -- aggregate_init_temp_controls
// pins the shapes that stay balanced.
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
