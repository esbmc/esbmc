// Non-vacuity guard for iterator_traits_tags: tag dispatch really resolves to
// the most derived overload, so expecting the wrong one must FAIL.
#include <iterator>
#include <cassert>

int which(std::input_iterator_tag)
{
  return 1;
}
int which(std::random_access_iterator_tag)
{
  return 4;
}

int main()
{
  assert(which(std::iterator_traits<int *>::iterator_category()) == 1);
  return 0;
}
