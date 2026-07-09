// Binding a const reference to a class prvalue (`const B& r = B(7)`) used to
// materialise the temporary and then copy it into a second temporary, so two
// B objects were constructed and both destroyed. The reference now binds
// directly to the single materialised temporary: one construction, one
// destructor, so g is 1 at the end of the scope.
#include <cassert>

int g;

struct B
{
  int x;
  B(int v) : x(v)
  {
  }
  ~B()
  {
    g++;
  }
};

int main()
{
  {
    const B &r = B(7);
    assert(r.x == 7);
  }
  assert(g == 1);
  return 0;
}
