test:
	@../test.pl -p -c esbmc

tests.log: ../test.pl
	@../test.pl -p -c esbmc

show:
	@for dir in *; do \
		if [ -d "$$dir" ]; then \
			vim -o "$$dir/*.c" "$$dir/*.out"; \
		fi; \
	done;
