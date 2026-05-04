// Issue #2040: std::vector<T>(n) must parse when T has only a
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
  std::vector<User> v(5);
  assert(v.size() == 5);
  return 0;
}
