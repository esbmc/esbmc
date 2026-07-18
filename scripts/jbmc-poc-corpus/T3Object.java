/* T3: object allocation and instance fields. */
public class T3Object {
  static class Point {
    int x;
    int y;

    Point(int x, int y) { this.x = x; this.y = y; }

    int normSquared() { return x * x + y * y; }
  }

  public static void main(String[] args) {
    Point p = new Point(3, 4);
    assert p.x == 3;
    assert p.normSquared() == 25;
    p.y = 0;
    assert p.normSquared() == 9;
  }
}
