// istream get
#include <iostream>
#include <fstream>
#include <cassert>
using namespace std;

int main () {
  char c, str[256];
  cin >> str;
  cout << cin.get();
  /*istream is;
  fstream file;
  
  file.open("example.txt", ios::in | ios::out);
  while(file.good()){
  	c = file.get();
  	if(file.good()){
  		file << "texto" << endl;
  	}
  
  }
  */
  return 0;
}
