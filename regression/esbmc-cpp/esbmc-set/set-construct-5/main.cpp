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

  set<int,classcomp> fifth;                 // class as Compare
  assert(fifth.size() == 0);
  bool(*fn_pt)(int,int) = fncomp;
  set<int,bool(*)(int,int)> sixth (fn_pt);  // function pointer as Compare

  return 0;
}
