fun main() {
    val arr = IntArray(20);
    for(i in 0..19)
      arr[i] = 20 - i;

    for(i in 0..19)
      assert(arr[i] == 20 - i);
}