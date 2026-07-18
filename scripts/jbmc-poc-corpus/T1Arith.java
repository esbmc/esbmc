/* T1: integer and boolean arithmetic. No allocation, no library calls. */
public class T1Arith {
  static int twice(int n) { return n + n; }

  public static void main(String[] args) {
    int x = 7;
    int d = twice(x);
    boolean even = (d % 2) == 0;
    assert d == 14;
    assert even;
    assert d > x;
  }
}
