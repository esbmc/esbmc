// A failing dynamic_cast<T&> throws std::bad_cast even when <typeinfo> is not
// included: the runtime constructs the object, so header visibility only governs
// whether the type can be *named* (catch(std::bad_cast&) / typeid). Here the
// cast fails and a catch(...) catches it, so the program is well-defined and
// must verify SUCCESSFUL. The lowering synthesizes a minimal std::bad_cast so it
// can materialize and throw a real object instead of falling back (#5075).
struct B
{
  virtual ~B()
  {
  }
};
struct D1 : B
{
};
struct D2 : B
{
};

int main()
{
  B *b = new D1();
  int caught = 0;
  try
  {
    D2 &d = dynamic_cast<D2 &>(*b); // fails: dynamic type is D1
    (void)d;
  }
  catch (...)
  {
    caught = 1;
  }
  __ESBMC_assert(caught == 1, "bad_cast caught by catch(...) without <typeinfo>");
  return 0;
}
