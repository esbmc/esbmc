// equal_range example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

bool mygreater (int i,int j) { return (i>j); }

int main () {
  int myints[] =  {10,20,30,30,20,10,10,20};
  int myints1[] = {10,10,10,20,20,20,30,30};
  int myints2[] = {30,30,20,20,20,10,10,10};
  vector<int> v(myints,myints+8);                         // 10 20 30 30 20 10 10 20
  pair<vector<int>::iterator,vector<int>::iterator> bounds;

  // using default comparison:
  v.assign(myints1,myints1+8);                       // 10 10 10 20 20 20 30 30
  bounds=equal_range (v.begin(), v.end(), 20);            //          ^        ^

  // using "mygreater" as comp:
  v.assign(myints2,myints2+8);                         // 30 30 20 20 20 10 10 10
  bounds=equal_range (v.begin(), v.end(), 20, mygreater); //       ^        ^
  assert((*bounds.first != 20) ||(*bounds.second != 10));
//  cout << "bounds at positions " << int(bounds.first - v.begin());
//  cout << " and " << int(bounds.second - v.begin()) << endl;

  return 0;
}
