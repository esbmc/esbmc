// set_union example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {
  int first[] = {5,10,15,20,25};
  int second[] = {10,20,30,40,50};
  
  vector<int> v(10);                           // 0  0  0  0  0  0  0  0  0  0
  vector<int>::iterator it;

 it = set_union (first, first+5, second, second+5, v.begin());
                                               // 5 10 15 20 25 30 40 50  0  0
  for(int i = 0;i < 10;i++) cout << v[i] << " ";
  cout << endl;
  assert(v[0] == 5);
  assert(v[1] == 10);
  assert(v[2] == 15);
  assert(v[3] == 20);
  assert(v[4] == 25);
  assert(v[5] == 30);
  assert(v[6] == 40);
  assert(v[7] == 50);
                                               
  return 0;
}
