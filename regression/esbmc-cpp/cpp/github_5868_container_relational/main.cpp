#include <list>
#include <set>
#include <map>
#include <cassert>

int main()
{
  // lexicographical, not by size: {1,3} < {2} in every ordered container.
  std::list<int> la, lb;
  la.push_back(1); la.push_back(3);
  lb.push_back(2);
  assert(la < lb);
  assert(!(lb < la));
  assert(la <= lb && lb > la && lb >= la && la != lb);

  std::set<int> sa, sb;
  sa.insert(1); sa.insert(3);
  sb.insert(2);
  assert(sa < sb);
  assert(!(sb < sa));
  assert(sa <= sb && sb > sa && sb >= sa && sa != sb);

  std::multiset<int> ma, mb;
  ma.insert(1); ma.insert(3);
  mb.insert(2);
  assert(ma < mb);
  assert(!(mb < ma));

  // a prefix is smaller
  std::set<int> p;
  p.insert(1);
  assert(p < sa);
  assert(!(sa < p));

  // map orders on the key first, the mapped value breaks a tie
  std::map<int, int> x, y;
  x.insert(std::pair<int, int>(1, 5));
  y.insert(std::pair<int, int>(1, 9));
  assert(x < y);
  assert(!(y < x));
  assert(x != y);
  y.clear();
  y.insert(std::pair<int, int>(2, 0));
  assert(x < y);

  std::map<int, int> z;
  z.insert(std::pair<int, int>(1, 5));
  assert(x == z);
  assert(x <= z && x >= z);

  std::multimap<int, int> qa, qb;
  qa.insert(std::pair<int, int>(1, 1));
  qb.insert(std::pair<int, int>(1, 2));
  assert(qa < qb);
  assert(qb > qa);
  return 0;
}
