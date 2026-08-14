/* A misaligned read from an array of pointers must be reported, exactly as
 * the int-array case in align-deref_fail is. It was not: the #4435 fix
 * declared such an access aligned whenever `alignment * 8 >= subtype_size`,
 * but `alignment` is already in bits by then, so the 8x over-approximation
 * suppressed check_alignment entirely. */
void *arr[16];
void *sink;

void f(unsigned k)
{
	k %= 8;
	/* 4-byte stride into 8-byte elements: every odd k is misaligned. */
	sink = *(void **)(void *)((int *)arr + k);
}

int main(void)
{
	f(3);
	return 0;
}
