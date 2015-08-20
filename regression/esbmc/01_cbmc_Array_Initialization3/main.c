unsigned char array[2][2] = {
  {1, 2},
  {3, 4}
};

void main(void)
{
  assert(array[0][1] ==2);
  assert(array[1][0] ==3);    // returned false in this case
}
