// Third round of STL gaps behind <boost/program_options.hpp>
// (github #6063): std::addressof, add_lvalue_reference, fpclassify,
// the [atomics.syn] fixed-width typedefs, and the
// basic_string iterator-pair constructor.
#include <memory>
#include <utility>
#include <cmath>
#include <atomic>
#include <string>
#include <cassert>

int main()
{
  int v = 42;
  assert(std::addressof(v) == &v);

  std::add_lvalue_reference<int>::type r = v;
  r = 7;
  assert(v == 7);

  assert(std::fpclassify(0.0) == FP_ZERO);
  assert(std::fpclassify(1.5) == FP_NORMAL);
  assert(std::fpclassify(1.0 / 0.0) == FP_INFINITE);

  std::atomic_int_least32_t counter(5);
  counter.store(9);
  assert(counter.load() == 9);

  const char *text = "hello";
  std::string s(text, text + 5);
  assert(s.length() == 5);
  assert(s[0] == 'h');
  assert(s[4] == 'o');

  return 0;
}
