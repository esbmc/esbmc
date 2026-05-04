// Distilled repro of issue #2040: a struct with a user-provided
// constructor (and therefore no implicit default constructor) used as
// the element type of std::vector, exercising push_back, range-for,
// and erase(remove_if(...), end()).
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>

struct User
{
  int id;
  std::string name;

  User(int id, const std::string &name) : id(id), name(name)
  {
  }
};

class UserManager
{
public:
  std::vector<User> users;

  void addUser(const User &u)
  {
    users.push_back(u);
  }

  User *getUserById(int id)
  {
    for (auto &u : users)
      if (u.id == id)
        return &u;
    return nullptr;
  }

  void removeUser(int id)
  {
    users.erase(
      std::remove_if(
        users.begin(), users.end(),
        [id](const User &u) { return u.id == id; }),
      users.end());
  }
};

int main()
{
  UserManager mgr;
  mgr.addUser(User(1, "Alice"));
  mgr.addUser(User(2, "Bob"));

  assert(mgr.users.size() == 2);
  assert(mgr.getUserById(1) != nullptr);
  assert(mgr.getUserById(3) == nullptr);

  mgr.removeUser(1);
  assert(mgr.users.size() == 1);
  assert(mgr.users[0].id == 2);
  return 0;
}
