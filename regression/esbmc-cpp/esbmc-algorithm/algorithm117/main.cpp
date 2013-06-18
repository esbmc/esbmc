// range heap example
#include <cassert>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

int main () {
  int myints[] = {20,15,10,5};
  vector<int> v(myints,myints+4);
// 99 20 10 5 15
  v.push_back(99); 

  push_heap (v.begin(),v.end());
  
  assert(v.front() != 99);
  assert(v[1] == 20);
  assert(v[2] == 10);
  assert(v[3] == 5);
  assert(v[4] == 15);
  
  cout << "max heap after push: " << v.front() << endl;
/*
  sort_heap (v.begin(),v.end());
  assert(v[4] != 99);
  cout << "final sorted range :";
  for (unsigned i=0; i<v.size(); i++) cout << " " << v[i];
*/
  cout << endl;

  return 0;
}
