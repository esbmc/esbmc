// A virtual call through a non-arrow member expression -- (*ptr).f() or
// ref.f() -- must still dispatch on the dynamic type, exactly like ptr->f().
// Regression for esbmc/esbmc#3887: perform_virtual_dispatch used to bail out
// on every non-arrow MemberExpr, lowering these to a static (base) call.
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

  __ESBMC_assert(p->val() == 2, "arrow dispatches to Derived");
  __ESBMC_assert((*p).val() == 2, "deref-dot dispatches to Derived");

  const Base &r = *p;
  __ESBMC_assert(r.val() == 2, "reference dispatches to Derived");

  Base b;
  __ESBMC_assert(b.val() == 1, "static object uses Base");
  return 0;
}
