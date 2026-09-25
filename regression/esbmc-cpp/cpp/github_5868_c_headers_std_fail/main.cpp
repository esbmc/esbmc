// Negative counterpart of github_5868_c_headers_std: the std:: names alias the
// real C functions rather than returning an unconstrained value, so a wrong
// claim about them is refuted.
#include <cctype>
#include <cassert>

int main()
{
  assert(std::isdigit('a'));
  return 0;
}
