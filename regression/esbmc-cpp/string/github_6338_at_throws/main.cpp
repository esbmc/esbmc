// KNOWNBUG (github #6338): std::string::at does not throw std::out_of_range.
//
// src/cpp/library/string still carries a pre-try-catch placeholder in
// basic_string::at -- `__ESBMC_assert(0, ...)` where the throw should be -- so
// correct exception-handling code is reported as VERIFICATION FAILED.
// std::vector::at throws correctly, which is the control below.
//
// The fix is blocked on a <string>/<stdexcept> include cycle: <stdexcept>
// includes <string> solely for the `const std::string &` constructor overloads
// of its exception classes, so out_of_range is incomplete at the point
// basic_string::at is parsed and a direct throw does not compile. Routing the
// throw through a helper compiles but defeats ESBMC's throw analysis, which
// only marks a function as throwing when the throw is directly in its body --
// that would make an *uncaught* string::at OOB verify SUCCESSFUL, which is
// unsound and worse than the current assert. See the issue for both directions.
//
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <string>
#include <vector>
#include <stdexcept>
#include <cassert>

int main()
{
  // Control: vector::at throws and the handler runs.
  std::vector<int> v;
  v.push_back(1);
  bool caught_vector = false;
  try
  {
    v.at(5);
  }
  catch (const std::out_of_range &)
  {
    caught_vector = true;
  }
  assert(caught_vector);

  // string::at should behave the same way.
  std::string s = "ab";
  bool caught_string = false;
  try
  {
    s.at(5);
  }
  catch (const std::out_of_range &)
  {
    caught_string = true;
  }
  assert(caught_string);

  return 0;
}
