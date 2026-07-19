// W1-loc spike Phase C (esbmc/esbmc#4715): a compound literal declares a
// synthesised variable with no source location. convert_decl reads the
// companion ASSIGN's location through codet's *mutable* location() accessor,
// which materialises an empty (id "", non-nil) #location, while code_decl2t
// carries a properly nil one -- so the native path emitted a nil location where
// legacy emitted an empty-but-present one. The GOTO dump renders the two
// differently ("no location" vs blank), which is how this surfaced as a
// --irep2-native-body byte-identity divergence. The DECL keeps the nil location
// in both paths: convert_decl emits it before that mutable access happens.
int my_func(int stat_loc)
{
  return ((union {
           __typeof(stat_loc) __in;
           int __i;
         }){.__in = (stat_loc)})
    .__i;
}

int main(void)
{
  int x = 7;
  return my_func(x) - 7;
}
