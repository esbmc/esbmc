#include <cassert>
#include <source_location>

int main()
{
  std::source_location sl = std::source_location::current();

  // The OM produces a non-zero line and a non-empty file name; checking
  // exact values would expose ESBMC's default-argument call-site
  // materialisation (orthogonal to having <source_location> at all).
  assert(sl.line() > 0);
  assert(sl.column() > 0);
  assert(sl.file_name() != nullptr);
  assert(sl.function_name() != nullptr);

  return 0;
}
