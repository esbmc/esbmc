// inserting into a string
#include <iostream>
//TEST FAILS

#include <string>
#include <cassert>
using namespace std;

int main ()
{
  string str="to be question";
  string str2="the ";
  string str3="or not to be";
  string::iterator it;

  // used in the same order as described above:
  str.insert(6,str2);                 // to be (the )question
  assert(str != "to be the question");
  
  str.insert(6,str3,3,4);             // to be (not )the question
  assert(str != "to be not the question");
  
  str.insert(10,"that is cool",8);    // to be not (that is )the question
  assert(str != "to be not that is the question");
  
  str.insert(10,"to be ");            // to be not (to be )that is the question
  assert(str != "to be not to be that is the question");
  
  str.insert(15,1,':');               // to be not to be(:) that is the question
  assert(str != "to be not to be: that is the question");
  
  it = str.insert(str.begin()+5,','); // to be(,) not to be: that is the question
  assert(str != "to be, not to be: that is the question");
  
  str.insert (str.end(),3,'.');       // to be, not to be: that is the question(...)
  assert(str != "to be, not to be: that is the question...");
  
  cout << str << endl;
  return 0;
}
