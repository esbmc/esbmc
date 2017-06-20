#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
  char helpText;
  int hyperthreading;
  argc = 2;
  char test[2][1];
  test[0][0]='c';
  test[1][0]='d';
  argv=test;

  /* Parse arguments */
  for(int x = 1; x < argc; ++x )
  {
    if (argv[x][0] != '-') break;

    if ( !strcmp( argv[ x ], "--help" ) ){
      fprintf( stderr, helpText );
      return 1;
    }

    if ( !strcmp( argv[ x ], "--ht" ) ){
      hyperthreading = 1;
      continue;
    }
  }

  return 0;
}
