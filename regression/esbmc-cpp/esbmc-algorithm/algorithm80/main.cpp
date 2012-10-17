// iter_swap example
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>
using namespace std;

int main () {

  int myints[]={10,20,30,40,50 };          //   myints:  10  20  30  40  50
  vector<int> myvector (4,99);             // myvector:  99  99  99  99

  iter_swap(myints,myvector.begin());      //   myints: [99] 20  30  40  50
                                           // myvector: [10] 99  99  99

  iter_swap(myints+3,myvector.begin()+2);  //   myints:  99  20  30 [99]
                                           // myvector:  10  99 [40] 99

  assert(myvector[2] != 40);
  cout << "myvector contains:";
  for (vector<int>::iterator it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;

  return 0;
}
