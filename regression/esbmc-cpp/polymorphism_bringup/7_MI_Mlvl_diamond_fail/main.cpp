/*
 * Multi-inheritance diamond problem
 * trivial constructor
 */
#include <cassert>

class File {
  public:
  virtual int f(void) { return 21; }
};

class InputFile: virtual public File {
  public:
    virtual int f(void) { return 42; }
};

class OutputFile: virtual public File {
  public:
    virtual int f(void) { return 63; }
};

class IOFile: public InputFile, public OutputFile {
  public:
    virtual int f(void) { return 52; }
};

int main(){
  IOFile *iofile = new IOFile();
  assert(iofile->File::f() == 21);
  assert(iofile->InputFile::f() == 42);
  assert(iofile->OutputFile::f() == 63);
  assert(iofile->f() == 51); // FAIL, should be 52
  delete iofile;
  return 0;
}


