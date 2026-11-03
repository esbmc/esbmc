#include <cassert>
struct E { virtual int id() const { return 1; } };
struct F : E { int id() const override { return 2; } };
struct G : F { int id() const override { return 3; } };
int main()
{
  int got = 0;
  try { throw G(); } catch (E e) { got = e.id(); }
  assert(got == 1);
  try { throw G(); } catch (F f) { got = f.id(); }
  assert(got == 2);
  try { throw F(); } catch (F f) { got = f.id(); }
  assert(got == 2);
  try { throw G(); } catch (E &e) { got = e.id(); }
  assert(got == 3);
  return 0;
}
