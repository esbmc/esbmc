// Two defects in <iterator>:
//
// 1. [iterator.traits] specifies iterator_traits as a struct, but it was
//    declared `class` with no access specifier, so all five member typedefs
//    were private. iterator_traits<It>::value_type on any class-type iterator
//    failed with "'value_type' is a private member"; only the T*
//    specialisation, already a struct, worked.
// 2. [iterator.tags] defines five categories. forward_iterator_tag and
//    bidirectional_iterator_tag did not exist, and random_access_iterator_tag
//    derived from nothing, so tag dispatch could not select an overload.
#include <iterator>
#include <cstddef>
#include <cassert>
#include <type_traits>

struct MyIt
{
  typedef std::ptrdiff_t difference_type;
  typedef int value_type;
  typedef int *pointer;
  typedef int &reference;
  typedef std::forward_iterator_tag iterator_category;
};

// tag dispatch must pick the more specialised overload
int which(std::input_iterator_tag)
{
  return 1;
}
int which(std::forward_iterator_tag)
{
  return 2;
}
int which(std::bidirectional_iterator_tag)
{
  return 3;
}
int which(std::random_access_iterator_tag)
{
  return 4;
}

int main()
{
  std::iterator_traits<MyIt>::value_type v = 7;
  assert(v == 7);
  assert((std::is_same<std::iterator_traits<MyIt>::pointer, int *>::value));
  assert((std::is_same<std::iterator_traits<int *>::value_type, int>::value));

  assert(which(std::iterator_traits<MyIt>::iterator_category()) == 2);
  assert(which(std::iterator_traits<int *>::iterator_category()) == 4);

  // [iterator.tags] inheritance
  assert((std::is_base_of<std::input_iterator_tag, std::forward_iterator_tag>::
            value));
  assert((std::is_base_of<
          std::forward_iterator_tag,
          std::bidirectional_iterator_tag>::value));
  assert((std::is_base_of<
          std::bidirectional_iterator_tag,
          std::random_access_iterator_tag>::value));
  return 0;
}
