#include <cassert>
struct T
{
  int v = 1;
  ~T() { v = 99; }
};
void f()
{
  T t;
  throw t.v;
}
int main()
{
  int got = 0;
  try { f(); } catch (int k) { got = k; }
  assert(got == 1);
  try { T t; throw t.v; } catch (int k) { got = k; }
  assert(got == 1);
  try { T t; throw t; } catch (T &e) { got = e.v; }
  assert(got == 1);
}
