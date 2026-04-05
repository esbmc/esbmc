/* Regression test: std::cin/cout/cerr must not trigger "extern variable not
 * found" warnings.  They are defined by compiling libstl.cpp in final(). */
#include <iostream>
#include <cassert>

int main()
{
  /* Just including iostream and using cout should not produce warnings. */
  std::cout << "hello";
  return 0;
}
