// #10786: the VLA size `0 & 0 / 0` is evaluated, so its division is checked.
int main()
{
  int z = 0;
  sizeof(int[3][0 & 1 / z]);
  sizeof(int[0 & 0 / 0]);
}
