union U { unsigned raw : 32; struct { unsigned x : 16, y : 16; }; };

int f();

int main() { (union U)f(); }
