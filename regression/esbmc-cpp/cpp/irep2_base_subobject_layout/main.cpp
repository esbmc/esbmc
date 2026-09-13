// Binding a reference to a non-primary base must displace onto that base's
// subobject. Under --clang-cpp-irep2-adjust-only the displacement comes from
// the IREP2 adjust pass, which reads a marker migrate_expr carries across the
// seam; dropped, `as.a` reads P's leading storage instead (#7025).
#include <cassert>

struct A
{
  int a;
  A() : a(1)
  {
  }
};

struct P
{
  virtual ~P()
  {
  }
  int p;
  P() : p(9)
  {
  }
};

struct AP : A, P
{
};

int main()
{
  AP ap;
  A &as = ap;
  assert(as.a == 1);
}
