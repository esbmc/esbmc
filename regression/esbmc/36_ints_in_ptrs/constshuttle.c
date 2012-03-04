int
main()
{
	void *face, *bees;
	int num;

	bees = &num;
	face = (void*)1234;
	num = (int)face;

	assert(num == 1234);
	return 0;
}
