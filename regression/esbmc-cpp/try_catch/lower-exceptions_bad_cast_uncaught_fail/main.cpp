// A failing dynamic_cast<T&> throws std::bad_cast; a handler for an unrelated
// type does not catch it, so it escapes -> terminate (uncaught).
#include <typeinfo>

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
    D2 &d = dynamic_cast<D2 &>(*b);
    (void)d;
  }
  catch (Other &)
  {
  }
  return 0;
}
