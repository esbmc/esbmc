// A failing dynamic_cast<T&> throws std::bad_cast; the exception lowering turns
// the __ESBMC_throw_bad_cast intrinsic into a real THROW and lowers it like any
// other, so the catch is reached.
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

int main()
{
  B *b = new D1();
  int caught = 0;
  try
  {
    D2 &d = dynamic_cast<D2 &>(*b);
    (void)d;
  }
  catch (std::bad_cast &)
  {
    caught = 1;
  }
  __ESBMC_assert(caught == 1, "bad_cast was caught");
  return 0;
}
