char global_buffer[64];

int main()
{
  __builtin_object_size(global_buffer, 0);
  return 0;
}
