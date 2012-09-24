/* memcmp example */
#include <iostream>
using std::cout;
using std::cin;
using std::endl;
#include <cstring>

int main ()
{
  char str1[256];
  char str2[256];
  int n;
  cout << "Enter a sentence: " << endl;
	cin >> str1; 
  cout << "Enter another sentence: " << endl;
	cin >> str2;
  n=memcmp ( str1, str2, 256 );
  if (n>0){
		cout << "'" << str1 << "' is greater than '" << str2 << "'." << endl;
	}else{
		if(n<0){
			cout << "'" << str1 <<"' is less than '" << str2 << "'." << endl;
		}else{
			cout << "'" << str1 <<"' is the same as '" << str2  << "'." << endl;
		}
	}
  return 0;
}
