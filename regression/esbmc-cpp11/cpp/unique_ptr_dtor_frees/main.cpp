// The unique_ptr models had no destructor: the scalar one was #if 0'd with a
// "fix remove goto sideeffect" TODO, and the array specialisation simply lacked
// one. The model therefore never freed, so --memory-leak-check reported a
// spurious "forgotten memory" leak on correct RAII.
#include <memory>
#include <cassert>

struct S
{
  int x;
  S() : x(0)
  {
  }
  S(int v) : x(v)
  {
  }
};

struct D
{
  void operator()(S *p) const
  {
    delete p;
  }
};

int main()
{
  {
    std::unique_ptr<S> p(new S(7));
    assert(p->x == 7);
  }

  {
    std::unique_ptr<S[]> a(new S[2]);
    a[1].x = 2;
    assert(a[1].x == 2);
  }

  {
    std::unique_ptr<S, D> c(new S(4), D());
    assert(c->x == 4);
  }

  {
    // Moving must not leave both objects owning the pointer.
    std::unique_ptr<S> m(new S(5));
    std::unique_ptr<S> n = std::move(m);
    assert(n->x == 5);
    assert(m.get() == nullptr);
  }

  {
    // release() hands ownership back: the caller frees, the model must not.
    std::unique_ptr<S> r(new S(6));
    S *raw = r.release();
    assert(raw->x == 6);
    delete raw;
  }

  return 0;
}
