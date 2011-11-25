SOURCES=$(ls -d */)

for i in $SOURCES
do
echo $i
#clang -emit-llvm $i/main.cpp -o $i/main.bc -c
#llc -march=c $i/main.bc -o $i/main.c
esbmc $i/main.c --unwind 10 --no-unwinding-assertions -I ~/libraries/
done
