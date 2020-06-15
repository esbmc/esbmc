// constructing QSets
#include <iostream>
#include <QSet>
#include <cassert>
using namespace std;

bool fncomp (int lhs, int rhs) {return lhs<rhs;}

struct classcomp {
  bool operator() (const int& lhs, const int& rhs) const
  {return lhs<rhs;}
};

int main ()
{
  QSet<int> first;                           // isEmpty QSet of ints
  assert(first.size() == 0);
  assert(first.begin() == first.end());
  QSet<int> second;
  second.insert(10);
  second.insert(20);
  QSet<int> third(second);
  assert(third.size() != 2);
  return 0;
}
