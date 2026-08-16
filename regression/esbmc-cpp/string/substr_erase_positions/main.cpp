#include <cassert>
#include <string>
int main() {
  std::string s = "hello";
  assert(s.substr(1,3)=="ell");
  assert(s.substr(1)=="ello");
  assert(s.substr(1,std::string::npos)=="ello");
  assert(s.substr(1,100)=="ello");
  assert(s.substr(5).empty());
  assert(s.substr(0)==s);

  std::string a = "abcd"; a.erase(2);
  assert(a=="ab");
  std::string b = "abcd"; b.erase(1,2);
  assert(b=="ad");
  std::string c = "abcd"; c.erase(0);
  assert(c.empty());
  std::string e = "abcd"; e.erase(4);
  assert(e=="abcd");
  std::string g = "abcd"; g.erase(1,100);
  assert(g=="a");
  return 0;
}
