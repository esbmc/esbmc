#include <cassert>
#include <utility>

template <class... T>
static constexpr std::size_t width(std::index_sequence_for<T...>)
{
  return sizeof...(T);
}

int main()
{
  // [intseq.general]: index_sequence_for<T...> is make_index_sequence<sizeof...(T)>.
  std::index_sequence_for<int, char, long> three;
  assert(three.size() == 3);

  std::index_sequence_for<> none;
  assert(none.size() == 0);

  std::index_sequence_for<int> one;
  assert(one.size() == 1);

  // It really is the same type as the sequence it aliases.
  std::make_index_sequence<3> made = three;
  assert(made.size() == 3);

  assert((width<int, char>(std::index_sequence_for<int, char>())) == 2);

  return 0;
}
