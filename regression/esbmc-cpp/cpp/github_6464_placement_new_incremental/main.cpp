// KNOWNBUG (github #6464): placement-new into malloc'd storage is reported as
// an out-of-bounds heap write under --incremental-bmc and --k-induction, while
// plain BMC (default, and every explicit --unwind N) proves the same program
// safe. Clean under clang++ -std=c++17 -fsanitize=address,undefined.
//
// Two ingredients are needed, both bisected on the issue: the placement-new
// itself (any plain assignment through the same address is fine, including via
// a void* round-trip across a function boundary), and a second malloc on a
// guarded path -- here grow(), which is never executed.
//
// This blocks #6368: constructing vector elements with placement-new is the
// right fix there, but it would make every growable std::vector<T> report a
// spurious failure in these two modes.
#include <new>
#include <cstdlib>

struct Elem
{
  char a[8];
};

struct Holder
{
  Elem *buf;
  int n;
  int cap;

  Holder()
  {
    buf = (Elem *)malloc(sizeof(Elem) * 10);
    if (!buf)
      abort();
    n = 0;
    cap = 10;
  }

  // Never executed below (n is 0, cap is 10), but its presence is what makes
  // the placement-new in add() report a spurious out-of-bounds heap write.
  void grow()
  {
    Elem *nb = (Elem *)malloc(sizeof(Elem) * 20);
    if (!nb)
      abort();
    buf = nb;
    cap = 20;
  }

  void add(const Elem &x)
  {
    if (n == cap)
      grow();
    new (buf + n) Elem(x);
    n++;
  }
};

int main()
{
  Holder h;
  Elem e;
  e.a[0] = 1;
  h.add(e);
  return 0;
}
