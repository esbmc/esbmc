// The argument temporary of a declaration's constructor dies at the end of the
// declaration ([class.temporary]/4), so a pointer kept from it dangles.
struct P
{
  int *p;
  P(int x) : p(new int(x))
  {
  }
  ~P()
  {
    delete p;
  }
};

struct Q
{
  int *q;
  Q(const P &x) : q(x.p)
  {
  }
};

int main()
{
  Q q(P(3));
  return *q.q;
}
