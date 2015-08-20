// lower_bound/upper_bound example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {
  int myints[] = {10,20,30,30,20,10,10,20};
  int myints1[] = {10,10,10,20,20,20,30,30};
  vector<int> v(myints,myints+8);           // 10 20 30 30 20 10 10 20
  vector<int>::iterator low,up;

  v.assign(myints1,myints1+8);              // 10 10 10 20 20 20 30 30

  low=lower_bound (v.begin(), v.end(), 20); //          ^
  assert(*low == 20);
//  up= upper_bound (v.begin(), v.end(), 20); //                   ^
//  assert(*up == 30);

  return 0;
}
