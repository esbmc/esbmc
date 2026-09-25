#include <list>
#include <iterator>
#include <cassert>

int main()
{
  typedef std::iterator_traits<std::list<int>::iterator> Tr;

  std::list<int> l;
  l.push_back(1);
  l.push_back(2);
  l.push_back(3);

  Tr::value_type sum = 0;
  for (std::list<int>::iterator i = l.begin(); i != l.end(); ++i)
  {
    Tr::reference r = *i;
    sum += r;
  }

  assert(sum == 7);
  return 0;
}
