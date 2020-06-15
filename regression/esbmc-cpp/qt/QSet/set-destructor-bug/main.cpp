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

  QSet<int> third (first);                  // a copy of second

  first.~QSet();
  third.~QSet();

  assert(first.size() != 0);

  return 0;
}
