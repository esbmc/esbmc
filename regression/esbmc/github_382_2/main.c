unsigned int global_var2 = 1;
unsigned int global_var3;

int main(void) {
  unsigned int *local_var_ptr = &global_var2;
  global_var3 = *local_var_ptr;
  return 0;
}
