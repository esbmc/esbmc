/*
 * multi-level inheritance, single column
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
  assert(iofile->f() != 10); // make sure it's calling the mid-level overriding function!
  assert(iofile->f() == 100);
  delete iofile;

  return 0;
}
