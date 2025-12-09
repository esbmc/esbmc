/*
 * Common error in multi-level single-column polymorphism
 */
#include <cassert>

class File
{
public:
  virtual int f(void)
  {
    return 1;
  }
  virtual int f(int)
  {
    return 2;
  }
};

class InputFile : public File
{
public:
  virtual int f(void)
  {
    return 10;
  }
  virtual int f(int)
  {
    return 20;
  }
};

class IOFile : public InputFile
{
public:
  virtual int f(void)
  {
    return 100;
  }
  virtual int f(int)
  {
    return 200;
  }
};

int main()
{
  File *iofile = new IOFile();
  assert(iofile->File::f() == 1);
  assert(iofile->File::f(1) == 2);
  assert(iofile->InputFile::f() == 10);  // ERROR
  assert(iofile->InputFile::f(1) == 20); // ERROR
  assert(iofile->f() == 100);
  assert(iofile->f(1) == 200);
  delete iofile;

  InputFile *iofile2 = new IOFile();
  assert(iofile2->File::f() == 1);
  assert(iofile2->InputFile::f() == 10);
  assert(iofile2->f() == 100);
  delete iofile2;

  return 0;
}
