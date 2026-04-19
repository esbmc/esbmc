void reach_error()
{
}

int main()
{
  double sum = 0.0;
  sum += 0.1;
  sum += 0.2;

  if (sum != 0.3)
    reach_error();
}
