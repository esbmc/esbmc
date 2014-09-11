// range heap example
#include <cassert>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

int main () {
  int myints[] = {30,20,10,5,15};
  vector<int> v(myints,myints+5);
  vector<int>::iterator it;
//20 15 10 5 30
//30 20 10 5 15

//  make_heap (v.begin(),v.end());
  cout << "initial max heap   : " << v.front() << endl;
  pop_heap (v.begin(),v.end()); 

  
  assert(v.front() == 20);
  assert(v[1] == 15);
  assert(v[2] == 10);
  assert(v[3] == 5);
  assert(v[4] != 30);
  
  v.pop_back();
  
/*
  v.push_back(99); push_heap (v.begin(),v.end());
  cout << "max heap after push: " << v.front() << endl;

  sort_heap (v.begin(),v.end());

  cout << "final sorted range :";
  for (unsigned i=0; i<v.size(); i++) cout << " " << v[i];
*/
  cout << endl;

  return 0;
}
