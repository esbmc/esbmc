char *p="abc";

int main()
{
  int input = -50;
  //char ch;

  /* should result in bounds violation */  
  //assert(input>=0 && input<3);
  p[input];
}
