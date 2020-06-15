//#include <stdio.h>
//#include <unistd.h>
#include <cassert>
#include <cstdio>
 #include <unistd.h>

class IOException
{
public:
    IOException () 
    {
      goto ERROR;
      ERROR:
        ;
    }
};

class MyFile
{
public:
  MyFile (const char *str)
  {
    this->fileName = str;
  }

  void readLine ()
  {
    if (getpid () % 2)
      throw IOException ();
  }

private:
  const char *fileName;

};

int
main ()
{
  try
    {
      MyFile *file = new MyFile ("sample");
      file->readLine ();
      delete file;
    } catch (IOException & e)
    {
      printf ("Some IO failed.\n");
    }
  return 0;
}
