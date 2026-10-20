#include "vars.h"
int nondet_int(void);
int engaged, throt;
unsigned char DSTATE, h;

/* Defect: throttle is not cut on the step cruise disengages. */
static void step(int in)
{
  engaged = in != 0;
  throt = engaged ? 30 : (DSTATE ? 33 : 0);
  DSTATE = engaged;
}

int main()
{
  for (int i = 0; i < 4; i++)
  {
    h = DSTATE;
    step(nondet_int());
  }
  return 0;
}
