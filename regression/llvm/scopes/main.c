float main()
{
  int b=0;

  b;

  if(b)
    assert(0);
  else
    2+2;

  if(b++) { 1 + 1; }

  int i=0;
  for(i++; i < 2; ++i)
    b++;

  for(int i = 0; i < 2; ++i) { b++; }

  while(b < 1)
    b--;

  do
    b--;
  while(b < 2);

  LABEL:
  LABEL1:
    b;
    b += 3;
    b -= 10;
    1+1;

    if(!b)
      goto LABEL;

  switch(b=3)
  {
    case 'c':
      ++b;
      break;

    case 2:
      b;
      break;

    default:
      assert(1);
  }

  return 2;
}
