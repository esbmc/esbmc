int main()
{
  int asd;

  for(int i=0;i<10000;i++){
    for(int i=0;i<10000;i++){};
    for(int i=0;i<10000;i++){};
    for(int i=0;i<10000;i++){};
    for(int i=0;i<10000;i++){};
  };

  while(1);

  __ESBMC_assert(0, "Error");
  return 0;
}