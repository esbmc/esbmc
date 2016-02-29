//#include <assert.h>

int k;

int fun8(float x, int y)
{
  int b=10;

  for (int i = 0; i < x; ++i)
  {
    int i;
    int x = i;
  }

  int i=5;
  for ( ; i < x; ++i)
  {
    assert(i == 5);
    int i=1;
    assert(i == 1);
    break;
    continue;
  }
  assert(i == 5);

  for (int i = 6; ; ++i)
  {
    assert(i == 6);
    int i=2;
    assert(i == 2);
    break;
  }
  assert(i == 5);

  for (int i = 7; i < x; )
  {
    assert(i == 7);
    int i=3;
    assert(i == 3);
    continue;
    break;
  }
  assert(i == 5);

  do {
    assert(i == 5);
    int i=4;
    assert(i == 4);
    break;
  } while(1);
  assert(i == 5);

  // New nested scope
  { 
    int b=100;

    for (int i = 0; i < x; ++i)
    {
      int i;
      int x = i;
    }

    int i=5;
    for ( ; i < x; ++i)
    {
      assert(i == 5);
      int i=1;
      assert(i == 1);
      break;
    }
    assert(i == 5);

    for (int i = 6; ; ++i)
    {
      assert(i == 6);
      int i=2;
      assert(i == 2);
      break;
    }
    assert(i == 5);

    for (int i = 7; i < x; )
    {
      assert(i == 7);
      int i=3;
      assert(i == 3);
      break;
    }
    assert(i == 5);

    do {
      assert(i == 5);
      int i=4;
      assert(i == 4);
      break;
    } while(1);
    assert(i == 5);


    assert(b == 100);
  }

  assert(b == 10);
  return b; 
}

int main()
{
  int x;
  x = fun8(2,-4);
  assert(x==10);
  return 0;
}

