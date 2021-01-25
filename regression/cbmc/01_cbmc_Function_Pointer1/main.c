int x = 0;

void f1(void) {
	x = 1;
}

void call(void (*f)()) {
	f();
}

int main() {
	call(f1);
}
