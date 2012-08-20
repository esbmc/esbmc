// generate algorithm example
#include <iostream>
#include <cassert>
#include <vector>
#include <ctime>
#include <cstdlib>
using namespace std;

template<class FwdIt, class Fn0>
void generate(FwdIt first, FwdIt last, Fn0 func) {
	while (first != last)
		*first++ = func();
}

// function generator:
int RandomNumber () { return (rand()%100); }

// class generator:
struct c_unique {
  int current;
  c_unique() {current=0;}
  int operator()() {return ++current;}
} UniqueNumber;


int main () {
  int i;
  srand ( unsigned ( time(NULL) ) );

  vector<int> myvector (8);
  vector<int>::iterator it;

  generate (myvector.begin(), myvector.end(), RandomNumber);

  cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  generate (myvector.begin(), myvector.end(), UniqueNumber);

  for (i=0;i<8;i++) assert(myvector[i] == ++i);

  cout << "\nmyvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    cout << " " << *it;

  cout << endl;
 
  return 0;
}
