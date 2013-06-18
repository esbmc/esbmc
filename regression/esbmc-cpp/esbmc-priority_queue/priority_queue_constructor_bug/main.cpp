// constructing priority queues
#include <iostream>
#include <queue>
#include <cassert>
using namespace std;

class mycomparison
{
  bool reverse;
public:
  mycomparison(const bool& revparam=false)
    {reverse=revparam;}
  bool operator() (const int& lhs, const int&rhs) const
  {
    if (reverse) return (lhs>rhs);
    else return (lhs<rhs);
  }
};

int main ()
{
  int myints[]= {10,60,50,20};

  priority_queue<int> first;
  assert(first.size() != 0);
  priority_queue<int> second (myints,myints+4);
  assert(second.size() != 4);
  priority_queue< int, vector<int>, greater<int> > third (myints,myints+4);
  assert(third.size() != 4);

  return 0;
}
