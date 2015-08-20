void pthread_exit(void *value_ptr);

int main()
{
  int i;

  if(i==100)
    pthread_exit(0);

  assert(i!=100);
}
