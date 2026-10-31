// #9353: the 65th distinct key was silently dropped, so this negated
// assertion was proved.
#include <cassert>
#include <unordered_set>

int main()
{
  std::unordered_set<int> s;
  for (int i = 0; i < 65; ++i)
    s.insert(i);
  assert(!(s.size() == 65));
  return 0;
}
