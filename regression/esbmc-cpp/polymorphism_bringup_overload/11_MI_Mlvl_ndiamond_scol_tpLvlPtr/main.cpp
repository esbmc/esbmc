/*
 * multi-level inheritance, single column
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
  assert(
    iofile->f() !=
    10); // make sure it's calling the mid-level overriding function!
  assert(
    iofile->f(1) !=
    20); // make sure it's calling the mid-level overriding function!
  assert(iofile->f() == 100);
  assert(iofile->f(1) == 200);
  delete iofile;

  return 0;
}
