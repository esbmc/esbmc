/* Regression test: std::cin/cout/cerr must not trigger "extern variable not
 * found" warnings.  They are silently nondet-initialized as known IO globals
 * (GitHub #3973). */
#include <iostream>

int main()
{
  return 0;
}
