// Non-vacuity guard for string_view_search: starts_with really computes, so
// asserting the negation of a true result must FAIL. Before the fix this test
// also failed -- but so did its positive counterpart, which is the tell-tale
// of a nondet return.
#include <string_view>
#include <cassert>

int main()
{
  std::string_view v("hello");
  assert(!v.starts_with(std::string_view("he")));
  return 0;
}
