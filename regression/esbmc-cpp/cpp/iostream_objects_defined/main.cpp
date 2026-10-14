// std::cin/cout/cerr were only DECLARED in <iostream> -- their definitions sit
// in libstl.cpp, which is mangled into the binary but never reaches the
// verified program. Their storage was therefore nondeterministic, so
// `std::cout.good()` satisfied neither itself nor its negation.
//
// ios::widen and ios::narrow had the same shape: declared, never defined, so
// widen(c) returned a nondeterministic char. [basic.ios.members] makes both the
// identity for a char stream.
#include <iostream>
#include <sstream>
#include <cassert>

int main()
{
  // The standard objects start in a good state.
  assert(std::cout.good());
  assert(!std::cout.fail());
  assert(!std::cout.bad());
  assert(!std::cout.eof());
  assert(std::cerr.good());
  assert(std::cin.good());

  assert(std::cout.widen('x') == 'x');
  assert(std::cout.narrow('x', '?') == 'x');

  // Locally constructed streams already worked; check they still do.
  std::ostringstream os;
  assert(os.good());

  // Setting a state bit is observable.
  std::ostringstream os2;
  os2.setstate(std::ios_base::failbit);
  assert(!os2.good());

  return 0;
}
