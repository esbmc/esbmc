#include <pthread.h>

a;
*d, *e;
b()
{
  if (a)
    ;
}
c()
{
  do
    ;
  while (a == 0);
}
main()
{
  pthread_create(&d, 0, b, 0);
  pthread_create(&e, 0, c, 0);
}
