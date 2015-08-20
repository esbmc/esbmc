default: tests.log

test:
	@./test.pl

tests.log: test.pl
	@./test.pl

show:
	@for dir in *; do \
		if [ -d "$$dir" ]; then \
			vim -o "$$dir/main.c" "$$dir/main.out"; \
		fi; \
	done;

clean:
	@for dir in *; do \
		if [ -d "$$dir" ]; then \
			rm -f "$$dir/main.out"; \
		fi; \
	done;
	@for dir in *; do \
		if [ -d "$$dir" ]; then \
			rm -f "$$dir/smt*.tmp"; \
		fi; \
	done;
	find . -name *.*~ | xargs rm -f
	find . -name *.out | xargs rm -f
	find . -name *.smt | xargs rm -f
	find . -name txt | xargs rm -f
	find . -name *.txt | xargs rm -f
	find . -name *.log | xargs rm -f
	find . -name *.trace | xargs rm -f
	rm -f tests.log	
