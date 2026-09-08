// The controls for aggregate_init_temp_double_destroy: constructing and
// copying the member directly stays balanced, and so does a member with no
// destructor, so the defect is specific to aggregate-initialising a wrapper
// from a temporary whose type has a destructor.
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

int plain_copies = 0;

struct N
{
  int v;
  N(int x) : v(x)
  {
  }
  N(const N &o) : v(o.v)
  {
    ++plain_copies;
  }
};

struct WN
{
  N n;
};

int main()
{
  {
    M a(5);
    M b = a;
    assert(b.v == 5);
  }
  assert(ctors == dtors);

  // a member without a destructor: the wrapper shape itself is fine
  WN w{N(7)};
  assert(w.n.v == 7);
  return 0;
}
