/* T1 negative: the final assertion is violated for every input. */
public class T1Arith_fail {
  static int twice(int n) { return n + n; }

  public static void main(String[] args) {
    int x = 7;
    int d = twice(x);
    assert d == 14;
    assert d == 15;
  }
}
