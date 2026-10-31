#include <cassert>
#include <vector>

int main()
{
  std::vector<int> v;
  v.reserve(10);
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  v.insert(v.begin(), v[1]);
  assert(v.size() == 4 && v[0] == 2 && v[1] == 1 && v[2] == 2 && v[3] == 3);

  std::vector<int> w(2);
  w[0] = 5;
  w[1] = 6;
  w.insert(w.begin(), w[1]);
  assert(w.size() == 3 && w[0] == 6 && w[1] == 5 && w[2] == 6);

  std::vector<int> u;
  u.reserve(10);
  u.push_back(7);
  u.push_back(8);
  u.insert(u.begin(), 2, u[1]);
  assert(u.size() == 4 && u[0] == 8 && u[1] == 8 && u[2] == 7 && u[3] == 8);
  return 0;
}
