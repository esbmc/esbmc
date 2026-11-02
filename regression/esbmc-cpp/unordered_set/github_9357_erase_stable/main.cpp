// #9357: erase invalidates only the erased element ([unord.req]).
#include <cassert>
#include <unordered_set>

int main()
{
  std::unordered_set<int> s;
  s.insert(1);
  s.insert(2);
  s.insert(3);
  auto it = s.find(2);
  s.erase(1);
  assert(*it == 2);

  int sum = 0;
  for (int x : s)
    sum += x;
  assert(sum == 5);

  s.erase(s.find(2));
  assert(s.size() == 1 && s.count(2) == 0);
  s.insert(4);
  assert(s.size() == 2 && s.count(3) == 1 && s.count(4) == 1);

  std::unordered_set<int> c = s;
  s.erase(s.begin(), s.end());
  assert(s.empty() && c.size() == 2 && c.count(4) == 1);

  std::unordered_set<int> w{1, 2, 3};
  w.erase(2);
  int keys = 0;
  for (int x : w)
    keys += x;
  assert(keys == 4);
  std::unordered_set<int> wc = w;
  assert(wc == w && wc.size() == 2);
  w.erase(w.begin(), w.find(3));
  assert(w.count(3) == 1 && w.count(2) == 0);
  return 0;
}
