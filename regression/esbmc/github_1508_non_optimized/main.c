float a;

void my_memset(char* ptr, int byte, int N)
{
  for(int i = 0; i < N; i++)
	  ptr[i] = byte;
}

main() {
  for (;;) {
    my_memset(&a, 0, sizeof(a));
    reach_error();
    for (;;)
      ;
  }
}
