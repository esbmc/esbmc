/*
 * Common error in multi-level single-column polymorphism
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
  assert(iofile->InputFile::f() == 10); // ERROR
  assert(iofile->f() == 100);
  delete iofile;

  InputFile *iofile2 = new IOFile();
  assert(iofile2->File::f() == 1);
  assert(iofile2->InputFile::f() == 10);
  assert(iofile2->f() == 100);
  delete iofile2;

  return 0;
}
