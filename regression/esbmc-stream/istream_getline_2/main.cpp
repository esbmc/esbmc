// istream getline
#include <iostream>
#include <cassert>
using namespace std;

int main () {
  char name[256], title[256];

  cout << "Enter your name (stop with percent) : ";
  cin.getline (name,256,'%');
  assert((int)cin.gcount() >= 0 && (int)cin.gcount() <= 256);
  cout << "Enter your favourite movie (stop with percent) : ";
  cin.getline (title,256, '%');
  assert((int)cin.gcount() >= 0 && (int)cin.gcount() <= 256);
  cout << name << "'s favourite movie is " << title;

  return 0;
}
