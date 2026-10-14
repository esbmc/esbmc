// github #6327: std::list front/back/rbegin/rend must be callable on a const
// list. The model declared only their mutating overloads, so const access
// failed to compile. Const overloads are added (front/back return a const
// reference; rbegin/rend mirror the existing const begin()/end()).
#include <cassert>
#include <list>

int endpoints(const std::list<int> &l)
{
  return l.front() + l.back();
}

int rsum(const std::list<int> &l)
{
  int s = 0;
  for (std::list<int>::const_reverse_iterator it = l.rbegin(); it != l.rend();
       ++it)
    s += *it;
  return s;
}

int main()
{
  std::list<int> l;
  l.push_back(1);
  l.push_back(2);
  l.push_back(3);

  const std::list<int> &cl = l;
  assert(endpoints(cl) == 4); // front 1 + back 3
  assert(*cl.rbegin() == 3);  // last element
  assert(rsum(cl) == 6);      // reverse traversal
  return 0;
}
