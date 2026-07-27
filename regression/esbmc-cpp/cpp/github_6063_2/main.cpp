// Second round of STL gaps behind <boost/program_options.hpp>
// (github #6063): allocator_traits, basic_ostream/basic_istream alias
// templates, std::streamsize, is_function / alignment_of / is_base_of.
#include <memory>
#include <ostream>
#include <istream>
#include <iostream>
#include <ios>
#include <type_traits>
#include <cassert>

void log_to(std::basic_ostream<char> &os)
{
}

struct B
{
};
struct D : B
{
};

int main()
{
  std::allocator<int> a;
  typedef std::allocator_traits<std::allocator<int>> AT;
  int *p = AT::allocate(a, 2);
  AT::construct(a, p, 41);
  *p += 1;
  assert(*p == 42);
  AT::destroy(a, p);
  AT::deallocate(a, p, 2);

  log_to(std::cout);
  std::streamsize n = 5;
  assert(n == 5);

  static_assert(std::is_function<void(int)>::value, "fn");
  static_assert(!std::is_function<int>::value, "nfn");
  static_assert(std::alignment_of<int>::value == alignof(int), "al");
  static_assert(std::is_base_of<B, D>::value, "bd");
  static_assert(!std::is_base_of<D, B>::value, "db");
  return 0;
}
