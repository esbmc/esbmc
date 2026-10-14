#include <string>
#include <cassert>

int main()
{
  // [string.op.plus]: the pointer operands are const charT*. They were
  // char*, so only a string literal bound -- through the conversion that
  // template deduction cannot apply to a const char* variable.
  std::string s("ab");
  const char *p = "cd";

  std::string right = s + p;
  assert(right.size() == 4);
  assert(right[2] == 'c');

  std::string left = p + s;
  assert(left.size() == 4);
  assert(left[0] == 'c');

  // A literal still works, and so does string + string.
  std::string lit = s + "ef";
  assert(lit.size() == 4);
  std::string both = s + s;
  assert(both.size() == 4);
  return 0;
}
