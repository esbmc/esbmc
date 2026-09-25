#include <string>

int main()
{
  // The null check must run before any dereference: with n == 0 nothing is
  // copied, so the only property that can fire is the null precondition. This
  // pins the argument order (previously strlen(s) was evaluated first and a
  // null s produced a misleading dereference failure inside strlen).
  const char *p = 0;
  std::string s(p, 0);

  return (int)s.length();
}
