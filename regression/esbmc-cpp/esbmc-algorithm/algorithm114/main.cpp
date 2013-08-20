// set_symmetric_difference example
#include <cassert>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

int main () {
  int first[] = {5,10,15,20,25};
  int second[] = {50,40,30,20,10};
  vector<int> v(10);                           // 0  0  0  0  0  0  0  0  0  0
  vector<int>::iterator it;

  sort (first,first+5);     //  5 10 15 20 25
  sort (second,second+5);   // 10 20 30 40 50

  it = set_symmetric_difference (first, first+5, second, second+5, v.begin());
                                               // 5 15 25 30 40 50  0  0  0  0
  assert(v[5] != 50);
//  cout << "symmetric difference has " << int(it - v.begin()) << " elements.\n";

  return 0;
}
