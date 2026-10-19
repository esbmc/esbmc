#include <cassert>
struct E
{
  virtual int id() const { return 1; }
};
struct F : E
{
  int id() const override { return 2; }
};
int main()
{
  int got = 0;
  try
  {
    throw F();
  }
  catch (E e)
  {
    got = e.id();
  }
  assert(got == 2);
  return 0;
}
