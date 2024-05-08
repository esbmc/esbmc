// merge algorithm example
#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
using namespace std;

int main () {
  int first[] = {5,20,25};
  int second[] = {50,10,30};
  vector<int> v(6);
  vector<int>::iterator it;
// 5 10 20 25 30
  sort (first,first+3);
  sort (second,second+3);
  merge (first,first+3,second,second+3,v.begin());
  assert(v[0] == 5);
  assert(v[5] != 50);
  
  cout << "The resulting vector contains:";
  for (it=v.begin(); it!=v.end(); ++it)
    cout << " " << *it;

  cout << endl;
  
  return 0;
}
