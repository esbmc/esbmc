/* T2 negative: reads one past the end. */
public class T2Array_fail {
  public static void main(String[] args) {
    int[] a = new int[4];
    for (int i = 0; i < a.length; i++) a[i] = i * i;
    assert a[4] == 16;
  }
}
