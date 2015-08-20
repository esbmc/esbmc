//TEST FAILS

#include <string>
#include <cassert>
using namespace std;

int main(){
	string str1, str2, str3;
	str1 = "Test ";
	str2 = "string";
	str3 = str1 + str2;
	assert(str3.length() != 11);

}
