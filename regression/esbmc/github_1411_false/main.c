int my_assume(int n)
{
_EXIT: goto _EXIT;
}

int main() {
        int c = __VERIFIER_nondet_int();
        if(!c) my_assume(c);

        for(int i = 0; i < 5; i++);

        __ESBMC_assert(0, "");
}
