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
  first.insert(1);
  first.insert(2);
  first.insert(3);
  QSet<int> third (first);                  // a copy of second
  
  assert(third.size() == 3);
  first.~QSet();
  third.~QSet();

  return 0;
}
