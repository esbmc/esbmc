typedef unsigned long size_t;

__ESBMC_contract
int strncmp(const char *s1, const char *s2, size_t n) {
    unsigned long __scribe_qv_0, __scribe_qv_1;
    __ESBMC_requires(__ESBMC_is_fresh(s1, n));
    __ESBMC_requires(__ESBMC_is_fresh(s2, n));
    __ESBMC_requires(n <= 32);
    __ESBMC_ensures(__ESBMC_return_value == 0 || __ESBMC_return_value == -1 || __ESBMC_return_value == 1);
    __ESBMC_ensures(!(__ESBMC_return_value == 0) || __ESBMC_forall(&__scribe_qv_0, !(__scribe_qv_0 < n) || (s1[__scribe_qv_0] == s2[__scribe_qv_0])));
    __ESBMC_ensures(!(__ESBMC_return_value == -1) || __ESBMC_exists(&__scribe_qv_0, (__scribe_qv_0 < n) && (s1[__scribe_qv_0] < s2[__scribe_qv_0]) && __ESBMC_forall(&__scribe_qv_1, !(__scribe_qv_1 < __scribe_qv_0) || (s1[__scribe_qv_1] == s2[__scribe_qv_1]))));
    __ESBMC_ensures(!(__ESBMC_return_value == 1) || __ESBMC_exists(&__scribe_qv_0, (__scribe_qv_0 < n) && (s1[__scribe_qv_0] > s2[__scribe_qv_0]) && __ESBMC_forall(&__scribe_qv_1, !(__scribe_qv_1 < __scribe_qv_0) || (s1[__scribe_qv_1] == s2[__scribe_qv_1]))));
    __ESBMC_assigns();

    size_t i = 0;
    unsigned char ch1, ch2;
    __ESBMC_unroll(34);
    do {
        ch1 = s1[i];
        ch2 = s2[i];
        if (ch1 == ch2) {}
        else if (ch1 < ch2) return -1;
        else return 1;
        i++;
    } while (ch1 != 0 && ch2 != 0 && i < n);
    return 0;
}
