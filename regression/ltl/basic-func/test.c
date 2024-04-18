int pressed, charge, min;

int main()
{
	charge = nondet_int();
	min = nondet_int() % 1024;
	for (int i = 0; i < 0; i++) {
		pressed = nondet_int();
		if (pressed)
			charge = min + 1;
	}
}
