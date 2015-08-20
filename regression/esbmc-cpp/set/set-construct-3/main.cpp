// constructing sets
#include <iostream>
#include <set>
#include <cassert>
using namespace std;

bool fncomp (int lhs, int rhs) {return lhs<rhs;}

struct classcomp {
  bool operator() (const int& lhs, const int& rhs) const
  {return lhs<rhs;}
};

int main ()
{
  set<int> first;                           // empty set of ints
  set<int>::iterator it;
  
  int myints[]= {10,20,30,40,50};
  set<int> second (myints,myints+5);        // pointers used as iterators
  
  set<int> third (second);                  // a copy of second
  assert(third.size() == 5);
  it = third.begin();
  assert(*it == 10);
  it = third.end();
  it--;
  assert(*it == 50);
  
  return 0;
}
