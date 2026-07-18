/* T3 negative: dereferences a null reference. */
public class T3Object_fail {
  static class Point {
    int x;

    Point(int x) { this.x = x; }
  }

  public static void main(String[] args) {
    Point p = new Point(3);
    Point q = null;
    assert p.x == 3;
    assert q.x == 3;
  }
}
