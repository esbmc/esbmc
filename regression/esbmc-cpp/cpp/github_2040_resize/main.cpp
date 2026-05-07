// Issue #2040: std::vector<T>::resize(n) must parse when T has only a
// user-provided constructor (no implicit default constructor).
#include <vector>
#include <string>
#include <cassert>

struct User
{
  int id;
  std::string name;

  User(int id, const std::string &name) : id(id), name(name)
  {
  }
};

int main()
{
  std::vector<User> v;
  v.push_back(User(1, "Alice"));
  v.resize(5);
  assert(v.size() == 5);
  assert(v[0].id == 1);
  return 0;
}
