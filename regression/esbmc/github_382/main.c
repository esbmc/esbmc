unsigned int global_var2 = 0;
unsigned int global_var3;

int main(void) {
  unsigned int *local_var_ptr = (unsigned int *)global_var2;
  global_var3 = *local_var_ptr;
  return 0;
}
