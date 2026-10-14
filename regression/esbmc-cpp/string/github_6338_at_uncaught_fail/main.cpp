// github #6338 (negative): an *uncaught* std::string::at out-of-range must be
// reported. Routing the throw through a helper would compile but defeat
// ESBMC's throw analysis, silently dropping the landing pad at the call site
// and verifying this program SUCCESSFUL -- which is what this test pins down.
//
// clang++ -std=c++17: terminates via std::terminate (exit != 0).
#include <string>

int main()
{
  std::string s = "ab";
  s.at(5);
  return 0;
}
