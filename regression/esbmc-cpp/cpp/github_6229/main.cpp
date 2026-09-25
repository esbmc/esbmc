#include <cassert>
#include <string>

int main()
{
  std::string a = "AAAA", b = "AA";

  // const char* on the left: was declared as a non-template friend but defined
  // as a template, so the call was undefined and returned a nondet result.
  assert("AAAA" > b);
  assert(!("AA" > a));
  assert("AA" < a);
  assert(!("AAAA" < b));

  // const char* on the right: a member operator>(const char*) and the free
  // operator>(basic_string&, const char*) both matched, so this did not compile.
  assert(a > "AA");
  assert(!(b > "AAAA"));
  assert(b < "AAAA");
  assert(!(a < "AA"));

  // Equal operands compare neither greater nor less in either direction.
  assert(!(a > "AAAA"));
  assert(!(a < "AAAA"));
  assert(!("AAAA" > a));
  assert(!("AAAA" < a));

  // >= / <= with the literal on the left: these free overloads were missing, so
  // "lit" >= s / "lit" <= s did not compile at all.
  assert("AAAA" >= b);
  assert("AAAA" >= a); // equal
  assert(!("AA" >= a));
  assert("AA" <= a);
  assert("AAAA" <= a); // equal
  assert(!("AAAA" <= b));

  return 0;
}
