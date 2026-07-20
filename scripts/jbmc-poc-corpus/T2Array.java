/* T2: array allocation, indexing and bounds. */
package jbmcpoc;

public class T2Array {
  public static void main(String[] args) {
    int[] a = new int[4];
    for (int i = 0; i < a.length; i++) a[i] = i * i;
    assert a.length == 4;
    assert a[3] == 9;
    assert a[0] == 0;
  }
}
