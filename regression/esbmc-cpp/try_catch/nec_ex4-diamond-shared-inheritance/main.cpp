#include <cstdio>

class streamexception
{
public:
  streamexception()
  {
    goto ERROR;
    ERROR:
    ;
  }
};
class inputstreamexception:public virtual streamexception
{
};
class outputstreamexception:public virtual streamexception
{
};
class iostreamexception:public inputstreamexception,
public outputstreamexception
{
};

int
nondet ()
{
  return 12 % 3;
}

int
iooperation ()
{
  return 0;
}

int
open_file (char *file, char *mode)
{
  int fd = 1;
  if (nondet ())
    throw streamexception ();
  else
    iooperation ();
  return fd;
}

int
write_file (int fd, char *str)
{
  if (nondet ())
    throw outputstreamexception ();
  else
    iooperation ();
  return 0;
}

char *
read_file (int fd)
{
  if (nondet ())
    throw inputstreamexception ();
  else
    iooperation ();
  return "random";
}

int
close_file (int fd)
{
  if (nondet ())
    throw streamexception ();
  else
    iooperation ();
  return 0;
}

int
copy_file (char *file1, char *file2)
{
  int fd1 = open_file (file1, "r");
  int fd2 = open_file (file2, "w");
  if (!nondet ())
    throw iostreamexception ();
  char *rstr = read_file (fd1);
  write_file (fd2, rstr);
  close_file (fd1);
  close_file (fd2);
}

int
fopen_file (char *file, char *mode)
{
  int fd = 1;
  iooperation ();
  return fd;
}

int
fwrite_file (int fd, char *str)
{
  iooperation ();
  return -1;
}

char *
fread_file (int fd)
{
  iooperation ();
  return "random";
}

int
fclose_file (int fd)
{
  iooperation ();
  return -1;
}

int
fcopy_file (int fd1, int fd2)
{
  if (fd1 != fd2)
    return -1;
  else
    return 4;
}

int
main ()
{
  char *file1 = "x";
  char *file2 = "y";
  char *file3 = "z";

  try
  {
    copy_file (file1, file2);
  }
  catch (iostreamexception & ioexcep)
  {
    printf ("Caught iostreamexception\n");
    goto ERROR;
    ERROR:
    ;
  }
  catch (inputstreamexception & iexcep)
  {
    printf ("Caught inputstreamexception\n");
  }
  catch (outputstreamexception & oexcep)
  {
    printf ("Caught outputstreamexception\n");
  }
  catch (streamexception & sexcep)
  {
    printf ("Caught streamexception\n");
  }

  try
  {
    int fd1 = fopen_file ("f1", "r");
    if (fd1 < 0)
    {
      inputstreamexception *tmp = new inputstreamexception ();
      streamexception & ex = *tmp;
      throw ex;
    }
    int fd2 = fopen_file ("f2", "w");
    if (fd2 < 0)
    {
      outputstreamexception *tmp = new outputstreamexception ();
      streamexception & ex = *tmp;
      throw ex;
    }
    int res = fcopy_file (fd1, fd2);
    if (res < 0)
    {
      streamexception *tmp = new iostreamexception ();
      streamexception & ex = *tmp;
      throw ex;
    }
    int f1 = fclose_file (fd1);
    if (f1 < 0)
    {
      streamexception *tmp = new streamexception ();
      streamexception & ex = *tmp;
      throw ex;
    }
    int f2 = fclose_file (fd2);
    if (f2 < 0)
    {
      streamexception *tmp = new streamexception ();
      streamexception & ex = *tmp;
      throw ex;
    }
  }
  catch (iostreamexception & ioex)
  {
    printf ("Caught iostreamexception\n");
  }
  catch (inputstreamexception & iex)
  {
    printf ("Caught inputstreamexception\n");
  }
  catch (outputstreamexception & oex)
  {
    printf ("Caught outputstreamexception\n");
  }
  catch (streamexception & sexcep)
  {
    printf ("Caught streamexception\n");
  }
}
