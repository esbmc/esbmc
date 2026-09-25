#include <cassert>
#include <vector>

// Exercises the branch shape the census observed at this handler: a branch
// whose native conversion leaks a scope-exit entry, reached through the
// container operational models. The delegated if-statement must roll back the
// abandoned attempt's temps and still produce the right values.
int main()
{
  std::vector<int> v;
  v.push_back(2);
  v.push_back(8);

  int total = 0;
  for (unsigned i = 0; i < v.size(); ++i)
    if (v[i] > 4)
      total += v[i];

  assert(v.size() == 2);
  assert(total == 10);
  return 0;
}
