#include <cassert>
#include <string>

int main()
{
  std::string a("a\0b", 3), b("a\0c", 3);
  assert(a.size() == 3 && a[2] == 'b');
  assert(a != b && !(a == b));
  assert(a < b && b > a && a <= b && b >= a);

  std::string p("a", 1), q("a\0", 2);
  assert(p < q && !(q < p) && p != q);

  std::string c("x\0y", 3);
  assert(c != "x" && !(c == "x") && c > "x" && "x" < c);
  assert(c >= "x" && "x" <= c);

  std::string d("abc");
  assert(d == "abc" && d == std::string("abc") && d < "abd");
  return 0;
}
