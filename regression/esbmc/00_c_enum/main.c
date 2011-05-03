typedef enum {
	fail,
	notfail,
	alsonotfail
} enumtype;

int
main()
{
	enumtype val;

	if (nondet_bool()) {
		val = notfail;
	} else {
		val = alsonotfail;
	}

	assert(val != fail);
	return 0;
}
