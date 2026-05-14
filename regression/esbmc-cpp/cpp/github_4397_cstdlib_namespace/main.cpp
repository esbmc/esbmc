#include <cstdlib>
#include <cassert>

int main()
{
  // <cstdlib> must expose the C library functions in namespace std.
  // Prior to the fix the only std:: names were strto*/aligned_alloc.
  int a = std::abs(-3);
  assert(a == 3);
  assert(std::labs(-5L) == 5L);

  // Just check std::div_t is a usable type name; std::div has no OM body
  // so we don't assert on its return value.
  std::div_t d;
  d.quot = 0;
  d.rem = 0;
  (void)d;

  if (a != 3)
    std::abort();

  // std::exit and std::atexit must also be visible.
  // We don't call exit() here so the assertion below is reachable.
  void (*p)(int) = &std::exit;
  (void)p;

  // std::getenv returns char* (may be NULL); just check the type compiles.
  const char *e = std::getenv("PATH");
  (void)e;

  return 0;
}
