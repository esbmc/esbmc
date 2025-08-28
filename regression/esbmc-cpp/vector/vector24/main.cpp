// vector::end
#include <iostream>
#include <vector>
#include <cassert>
using namespace std;

// g++ -g -O1 -fsanitize=address -fno-omit-frame-pointer main.cpp -o test_asan
// ./test_asan
// test_asan: main.cpp:15: int main(): Assertion `*it == 0' failed.
// Aborted (core dumped)

int main ()
{
  vector<int> myvector;
  for (int i=1; i<=5; i++) myvector.insert(myvector.end(),i);

  cout << "myvector contains:";
  vector<int>::iterator it;
  it=myvector.end();
  assert(*it == 0);
  for ( it=myvector.begin() ; it < myvector.end(); it++ )
    cout << " " << *it;

  cout << endl;

  return 0;
}
