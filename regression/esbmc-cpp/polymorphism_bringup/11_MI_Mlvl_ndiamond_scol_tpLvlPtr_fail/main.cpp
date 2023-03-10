/*
 * multi-level inheritance, single column
 * polymorphism using top-level pointer
 */
#include <cassert>

class File {
  public:
  virtual int f(void) { return 1; }
};

class InputFile: public File {
  public:
    virtual int f(void) { return 10; }
};

class IOFile: public InputFile {
  public:
    virtual int f(void) { return 100; }
};

int main(){
  File *iofile = new IOFile();
  assert(iofile->File::f() == 1);
  assert(iofile->f() == 10); // this should not pass
  delete iofile;

  return 0;
}
