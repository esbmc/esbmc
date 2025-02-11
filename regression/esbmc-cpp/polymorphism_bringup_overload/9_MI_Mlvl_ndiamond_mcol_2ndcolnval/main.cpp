/*
 * multiple inheritance but NOT diamond problem, late binding
 */
#include <cassert>

class File
{
public:
  virtual int f(void)
  {
    return 21;
  }
  virtual int f(int)
  {
    return 22;
  }
};

class InputFile : virtual public File
{
public:
  virtual int f(void)
  {
    return 42;
  }
  virtual int f(int)
  {
    return 43;
  }
};

class OutputFile
{
public:
  int f(void)
  {
    return 63;
  }
  int f(int)
  {
    return 64;
  }
};

class IOFile : public InputFile, public OutputFile
{
public:
  virtual int f(void)
  {
    return 52;
  }
  virtual int f(int)
  {
    return 53;
  }
};

int main()
{
  IOFile *iofile = new IOFile();
  assert(iofile->File::f() == 21);
  assert(iofile->File::f(1) == 22);
  assert(iofile->InputFile::f() == 42);
  assert(iofile->InputFile::f(1) == 43);
  assert(iofile->OutputFile::f() == 63);
  assert(iofile->OutputFile::f(1) == 64);
  assert(iofile->f() == 52);
  assert(iofile->f(1) == 53);
  delete iofile;
  return 0;
}
