float MaxCost = 100;
unsigned char PanelQuant = 4;
float PanelData[4][13];
unsigned char nondet_uchar();

int Faux(float cost)
{
  unsigned char PanelChoice;
  PanelChoice = nondet_uchar();
  __VERIFIER_assume(PanelChoice <= 3);
  float Ap = PanelData[PanelChoice][0];
  return 0;
}

int main()
{
  float HintCost;
  for(HintCost = 0; HintCost <= MaxCost; HintCost++)
  {
    Faux(HintCost);
  }
  return 0;
}
