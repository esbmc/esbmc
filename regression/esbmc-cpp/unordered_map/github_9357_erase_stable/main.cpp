// #9357: erase invalidates only the erased element ([unord.req]).
#include <cassert>
#include <unordered_map>

int main()
{
  std::unordered_map<int, int> m;
  m[1] = 10;
  m[2] = 20;
  m[3] = 30;
  auto it = m.find(2);
  int &r = m[2];
  m.erase(1);
  assert(it->first == 2 && it->second == 20);
  assert(r == 20);

  int sum = 0;
  for (const auto &p : m)
    sum += p.second;
  assert(sum == 50);

  m.erase(m.find(2));
  assert(m.size() == 1 && m.count(2) == 0);
  m[4] = 40;
  assert(m.size() == 2 && m.at(3) == 30 && m.at(4) == 40);

  std::unordered_map<int, int> c = m;
  m.erase(m.begin(), m.end());
  assert(m.empty() && c.size() == 2 && c.at(4) == 40);
  return 0;
}
