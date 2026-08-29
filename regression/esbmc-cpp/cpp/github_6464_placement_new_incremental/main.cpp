// github #6464: placement-new into malloc'd storage was reported as an
// out-of-bounds heap write under --incremental-bmc and --k-induction, but only
// when a second malloc sits on a guarded path. Clean under clang++ -std=c++17
// -fsanitize=address,undefined.
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

  // Never executed (n is 0, cap is 10); its mere presence is what triggered
  // the false positive.
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
