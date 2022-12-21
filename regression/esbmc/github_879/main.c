int main()
{
    float r[1][1];
    (*(&r))[0][0] = nondet();
    assert(!(*(&r))[0][0]);
    return 0;
}

