// A failing dynamic_cast<T&> throws std::bad_cast, which is caught here. This
// requires symex to resolve the <typeinfo> model's std::bad_cast symbol, whose
// tag name is elaborated ("class std::bad_cast") on newer Clang/LLVM.
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
  try
  {
    D2 &d = dynamic_cast<D2 &>(*b); // dynamic type is D1, not D2 -> bad_cast
    (void)d;
    return 2; // not reached
  }
  catch (std::bad_cast &)
  {
    return 1;
  }
  return 0;
}
