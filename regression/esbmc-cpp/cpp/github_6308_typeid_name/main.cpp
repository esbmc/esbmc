// github #6308: type_info::name() must return a null-terminated string, so C
// string operations on it are well-defined. The name array was sized without a
// slot for the terminating '\0', so strlen ran off the end.
#include <cassert>
#include <typeinfo>
#include <cstring>

struct A {};
struct P { virtual ~P() {} };

int main()
{
  assert(strlen(typeid(int).name()) > 0);
  assert(strlen(typeid(A).name()) > 0);

  P p;
  assert(strlen(typeid(p).name()) > 0);

  const char *n = typeid(double).name();
  assert(n[0] != '\0'); // first byte readable, string non-empty
  return 0;
}
