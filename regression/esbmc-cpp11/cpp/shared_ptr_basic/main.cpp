// github #5868 gap 4: <memory> modelled allocator/unique_ptr/make_unique but
// never declared shared_ptr, so any use failed with "no template named
// 'shared_ptr' in namespace 'std'". 40 of ESBMC's own source files use it.
#include <memory>
#include <cassert>

struct S
{
  int x;
  S(int v) : x(v)
  {
  }
};

struct B
{
  virtual ~B()
  {
  }
  int b;
  B() : b(1)
  {
  }
};
struct D : B
{
  int d;
  D() : d(2)
  {
  }
};

int main()
{
  {
    std::shared_ptr<S> a(new S(7));
    assert(a->x == 7);
    assert(a.use_count() == 1);
    assert(a.unique());
    {
      std::shared_ptr<S> b = a;
      assert(a.use_count() == 2);
      assert(b.get() == a.get());
      assert(a == b);
    }
    assert(a.use_count() == 1); // the copy released its share
  }

  {
    std::shared_ptr<S> e;
    assert(!e);
    assert(e.use_count() == 0);
    assert(e == nullptr);
  }

  {
    std::shared_ptr<S> m = std::make_shared<S>(9);
    assert(m->x == 9);
    assert(m.use_count() == 1);
  }

  {
    std::shared_ptr<S> a(new S(4));
    std::shared_ptr<S> b = std::move(a);
    assert(a.get() == nullptr);
    assert(b->x == 4);
    assert(b.use_count() == 1);
  }

  {
    // assignment releases the old object
    std::shared_ptr<S> a(new S(5));
    std::shared_ptr<S> b(new S(6));
    a = b;
    assert(a->x == 6);
    assert(b.use_count() == 2);
  }

  {
    std::shared_ptr<S> a(new S(8));
    a.reset();
    assert(!a);
    a.reset(new S(10));
    assert(a->x == 10);
    std::shared_ptr<S> b;
    a.swap(b);
    assert(!a);
    assert(b->x == 10);
  }

  {
    std::shared_ptr<D> d(new D());
    std::shared_ptr<B> b = d; // Derived* -> Base*
    assert(b->b == 1);
    assert(d.use_count() == 2);
  }

  return 0;
}
