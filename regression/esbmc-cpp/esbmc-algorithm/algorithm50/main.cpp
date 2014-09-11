// inplace_merge example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {
  int first[] = {5,10,15,20,25};
  int second[] = {10,20,30,40,50};
  vector<int> v(10);
  vector<int>::iterator it;
// 5 10 10 15 20 20 25 30 40 50
  copy (first,first+5,v.begin());
  copy (second,second+5,v.begin()+5);
  
  inplace_merge (v.begin(),v.begin()+5,v.end());
  
  assert(v[0] == 5);
  assert(v[1] == 10);
  assert(v[2] == 10);
  assert(v[3] == 15);
  assert(v[4] == 20);
  assert(v[5] == 20);
  assert(v[6] == 25);
  assert(v[7] == 30);
  assert(v[8] == 40);
  assert(v[9] == 50);

  cout << "The resulting vector contains:";
  for (it=v.begin(); it!=v.end(); ++it)
    cout << " " << *it;

  cout << endl;
  
  return 0;
}
