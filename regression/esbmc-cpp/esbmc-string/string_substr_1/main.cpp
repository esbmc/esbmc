#include <string>
#include <cassert>
#include <iostream>

using namespace std;

int main(){
	string str1, str2, str3;
	
	str1 = string("Test string test");
	str2 = str1.substr(5,6);
	cout << str2;
	assert(str2 == "string");
}
