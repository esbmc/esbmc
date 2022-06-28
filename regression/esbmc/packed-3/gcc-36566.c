/* slightly adapted from <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=36566> */

struct S
{
    short   s;
} __attribute__((aligned(8), packed));

void fun3(struct S *s)
{
	/* KNOWNBUG: this should not warn about not checking the alignment
	 * and actually check the alignment as member `S::s` is actually
	 * known to be aligned. */
	s->s;
}
