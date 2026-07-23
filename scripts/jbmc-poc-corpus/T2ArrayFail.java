/* T2 negative: reads one past the end. */
package jbmcpoc;

public class T2ArrayFail {
  public static void main(String[] args) {
    int[] a = new int[4];
    for (int i = 0; i < a.length; i++) a[i] = i * i;
    assert a[4] == 16;
  }
}
