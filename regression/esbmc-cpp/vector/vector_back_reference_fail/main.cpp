// [sequence.reqmts] gives non-const back() the return type reference, so the
// last element is writable in place (#7537).
#include <cassert>
#include <vector>

int main()
{
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  v.back() = 42;
  assert(v.back() == 42);
  assert(v[2] == 43);

  int &r = v.back();
  r = 7;
  assert(v[2] == 7);

  v.front() = 5;
  assert(v[0] == 5);

  return 0;
}
