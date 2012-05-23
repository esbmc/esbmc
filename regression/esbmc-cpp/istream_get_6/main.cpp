// istream get
#include <iostream>
#include <fstream>
#include <streambuf>
#include <cassert>
using namespace std;

int main () {
  //char c, str[256];
  streambuf sb;
  
  cin.get(sb, '%');  
  
  
  return 0;
}
