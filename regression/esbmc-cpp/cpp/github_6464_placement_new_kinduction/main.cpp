// github #6464 (k-induction): the same program under the other mode the
// defect hit; see github_6464_placement_new_incremental.
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
