// Negative counterpart to virtual_dispatch_deref_dot: a non-arrow virtual call
// must select the dynamic override, so asserting the Base result on an object
// whose dynamic type is Derived must FAIL. Guards against a fix that merely
// silences the property instead of dispatching correctly.
struct Base
{
  virtual ~Base() = default;
  virtual int val() const
  {
    return 1;
  }
};

struct Derived : Base
{
  int val() const override
  {
    return 2;
  }
};

int main()
{
  Derived d;
  Base *p = &d;
  const Base &r = *p;
  __ESBMC_assert(r.val() == 1, "must fail: reference dispatches to Derived");
  return 0;
}
