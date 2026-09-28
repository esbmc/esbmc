#include <list>
#include <iterator>
#include <type_traits>
#include <cassert>

int main()
{
  typedef std::list<int>::iterator It;
  typedef std::iterator_traits<It> Tr;

  static_assert(std::is_same<Tr::value_type, int>::value, "value_type");
  static_assert(std::is_same<Tr::reference, int &>::value, "reference");
  static_assert(std::is_same<Tr::pointer, int *>::value, "pointer");
  static_assert(
    std::is_same<Tr::iterator_category, std::bidirectional_iterator_tag>::value,
    "a list iterator is bidirectional, not random-access");

  std::list<int> l;
  l.push_back(1);
  l.push_back(2);
  l.push_back(3);

  // Goes through the bidirectional arm of distance's tag dispatch.
  assert(std::distance(l.begin(), l.end()) == 3);
  return 0;
}
