#ifndef uint16_t
#define uint16_t unsigned short
#endif
int main()
{
  uint16_t a = 0xFFFF;
  uint16_t b = a << 16;
  return 0;
}