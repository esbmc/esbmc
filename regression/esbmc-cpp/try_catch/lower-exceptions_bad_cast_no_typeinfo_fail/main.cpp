// A failing dynamic_cast<T&> throws std::bad_cast even without <typeinfo>. Here
// only an unrelated handler is present, so the bad_cast is not caught: it escapes
// to std::terminate, which the lowering reports as an uncaught exception ->
// VERIFICATION FAILED. The std::bad_cast type is synthesized (no <typeinfo>), so
// the lowering throws a real object rather than falling back (#5075).
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
struct Other
{
};

int main()
{
  B *b = new D1();
  try
  {
    D2 &d = dynamic_cast<D2 &>(*b); // fails: dynamic type is D1
    (void)d;
  }
  catch (Other &)
  {
  }
  return 0;
}
