#include <map>
#include <cstring>
#include <cassert>
using namespace std;

class StringClass {
  char str[20];
public:
  StringClass() { strcpy(str, ""); }
  StringClass(char *s) { strcpy(str, s); }
  char *get() { return str; }
};
bool operator<(StringClass a, StringClass b) { return strcmp(a.get(), b.get()) < 0; }

int main() {
  multimap<StringClass, int> m;
  m.insert(multimap<StringClass, int>::value_type(StringClass("y"), 1));
  assert(m.size() == 1);   // holds
  return 0;
}
