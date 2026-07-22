#include <cassert>
#include <string>

int main()
{
  std::string a = "AAAA", b = "AA";

  // Literal-on-the-left overloads must be deterministic: the in-class friend
  // declared a non-template function that the out-of-line template definition
  // never matched, so ESBMC modelled the call as an undefined (nondet) function.
  bool lt1 = ("AAAA" > b);
  bool lt2 = ("AAAA" > b);
  assert(lt1 == lt2);
  assert("AAAA" > b);
  assert(!("AA" > a));
  assert(!("AAAA" < b));
  assert("AA" < a);

  // Literal-on-the-right overloads must even compile: a member
  // operator>(const char*) and a free operator>(basic_string&, const char*)
  // both matched, making every such call ambiguous.
  assert(a > "AA");
  assert(!(b > "AAAA"));
  assert(!(a < "AA"));
  assert(b < "AAAA");

  return 0;
}
