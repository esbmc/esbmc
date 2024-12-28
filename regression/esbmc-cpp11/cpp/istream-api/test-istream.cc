
#include <istream>
#include <iostream>

std::ios & g(std::ios &i)
{
	return i;
}

void f(std::istream &is)
{
	is >> std::boolalpha >> std::noboolalpha;
	is >> std::showbase >> std::noshowbase;
	is >> std::showpos >> std::noshowpos;
	is >> std::skipws >> std::noskipws;
	is >> std::uppercase >> std::nouppercase;
	is >> std::unitbuf >> std::nounitbuf;
	is >> std::internal >> std::left >> std::right;
	is >> std::oct >> std::hex >> std::dec;
	is >> std::hexfloat >> std::defaultfloat;
	is >> std::fixed >> std::scientific;

	is >> g;

	bool b;
	char c;
	signed char sc;
	unsigned char uc;
	short s;
	unsigned short us;
	int i;
	unsigned int ui;
	long l;
	unsigned long ul;
	long long ll;
	unsigned long long ull;
	void *p;
	float f;
	double d;
	long double ld;

	is >> b;

	is >> c;
	is >> sc;
	is >> uc;

	is >> s;
	is >> us;
	is >> i;
	is >> ui;
	is >> l;
	is >> ul;

	is >> ll;
	is >> ull;

	is >> p;

	is >> f;
	is >> d;
	is >> ld;
}

int main() {
  f(std::cin);
}