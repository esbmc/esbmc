// github #5868 gap 4: weak_ptr must observe the pointee's death while keeping
// the control block alive, which is what the second (weak) count is for.
#include <memory>
#include <cassert>

struct S
{
  int x;
  S(int v) : x(v)
  {
  }
};

int main()
{
  std::weak_ptr<S> w;
  assert(w.expired());

  {
    std::shared_ptr<S> a(new S(3));
    w = a;
    assert(!w.expired());
    assert(w.use_count() == 1);

    std::shared_ptr<S> l = w.lock();
    assert(l.get() == a.get());
    assert(a.use_count() == 2); // lock() took a share
  }

  assert(w.expired());
  assert(w.use_count() == 0);
  assert(w.lock() == nullptr);

  return 0;
}
