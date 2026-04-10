# Comprehensive test for passing objects with forward-referenced types - FAIL cases
# Tests pointer-to-struct semantics for Python objects passed as parameters
# All assertions are intentionally incorrect to verify proper failure detection

# Test 1: Basic object passing with single attribute
class Point:
    def __init__(self, x: int) -> None:
        self.x: int = x
    
    def create_vector(self) -> 'Vector':
        return Vector(self)

class Vector:
    def __init__(self, p: Point) -> None:
        self.x: int = p.x  # Access attribute from parameter object

# Test basic object passing - FAIL
p1: Point = Point(10)
v1: Vector = p1.create_vector()
assert v1.x == 99  # Wrong: should be 10


# Test 2: Object passing with multiple attributes
class Rectangle:
    def __init__(self, width: int, height: int) -> None:
        self.width: int = width
        self.height: int = height
    
    def to_box(self) -> 'Box':
        return Box(self)

class Box:
    def __init__(self, rect: Rectangle) -> None:
        self.width: int = rect.width
        self.height: int = rect.height
        self.area: int = rect.width * rect.height

rect: Rectangle = Rectangle(5, 8)
box: Box = rect.to_box()
assert box.width == 8  # Wrong: should be 5
assert box.height == 5  # Wrong: should be 8
assert box.area == 100  # Wrong: should be 40


class DataA:
    def __init__(self, value: int) -> None:
        self.value: int = value
    
    def to_b(self) -> 'DataB':
        return DataB(self)

class DataB:
    def __init__(self, a: DataA) -> None:
        self.value: int = a.value
        self.doubled: int = a.value * 2
    
    def to_c(self) -> 'DataC':
        return DataC(self)

class DataC:
    def __init__(self, b: DataB) -> None:
        self.original: int = b.value
        self.doubled: int = b.doubled
        self.tripled: int = b.value * 3

data_a: DataA = DataA(7)
data_b: DataB = data_a.to_b()
assert data_b.value == 14  # Wrong: should be 7
assert data_b.doubled == 7  # Wrong: should be 14

data_c: DataC = data_b.to_c()
assert data_c.original == 21  # Wrong: should be 7
assert data_c.doubled == 21  # Wrong: should be 14
assert data_c.tripled == 14  # Wrong: should be 21


# Test 4: Multiple parameter objects
class Source1:
    def __init__(self, x: int) -> None:
        self.x: int = x

class Source2:
    def __init__(self, y: int) -> None:
        self.y: int = y

class Merger:
    def __init__(self, s1: Source1, s2: Source2) -> None:
        self.x: int = s1.x
        self.y: int = s2.y
        self.sum: int = s1.x + s2.y

src1: Source1 = Source1(10)
src2: Source2 = Source2(20)
merged: Merger = Merger(src1, src2)
assert merged.x == 20  # Wrong: should be 10
assert merged.y == 10  # Wrong: should be 20
assert merged.sum == 0  # Wrong: should be 30


# Test 5: Object with computed attributes
class Temperature:
    def __init__(self, celsius: int) -> None:
        self.celsius: int = celsius
    
    def to_converter(self) -> 'TempConverter':
        return TempConverter(self)

class TempConverter:
    def __init__(self, temp: Temperature) -> None:
        self.celsius: int = temp.celsius
        self.fahrenheit: int = (temp.celsius * 9) // 5 + 32  # Integer division for simplicity

temp: Temperature = Temperature(0)
converter: TempConverter = temp.to_converter()
assert converter.celsius == 100  # Wrong: should be 0
assert converter.fahrenheit == 0  # Wrong: should be 32

temp2: Temperature = Temperature(100)
converter2: TempConverter = temp2.to_converter()
assert converter2.celsius == 0  # Wrong: should be 100
assert converter2.fahrenheit == 32  # Wrong: should be 212


# Test 6: Nested object creation with method chains
class Node:
    def __init__(self, val: int) -> None:
        self.val: int = val
    
    def create_wrapper(self) -> 'Wrapper':
        return Wrapper(self)

class Wrapper:
    def __init__(self, node: Node) -> None:
        self.val: int = node.val
        self.wrapped_val: int = node.val + 100
    
    def create_container(self) -> 'Container':
        return Container(self)

class Container:
    def __init__(self, wrapper: Wrapper) -> None:
        self.original: int = wrapper.val
        self.wrapped: int = wrapper.wrapped_val
        self.total: int = wrapper.val + wrapper.wrapped_val

node: Node = Node(5)
wrapper: Wrapper = node.create_wrapper()
assert wrapper.val == 105  # Wrong: should be 5
assert wrapper.wrapped_val == 5  # Wrong: should be 105

container: Container = wrapper.create_container()
assert container.original == 105  # Wrong: should be 5
assert container.wrapped == 5  # Wrong: should be 105
assert container.total == 0  # Wrong: should be 110


# Test 7: Object passing with parameter validation
class Positive:
    def __init__(self, value: int) -> None:
        self.value: int = value
    
    def to_doubled(self) -> 'Doubled':
        return Doubled(self)

class Doubled:
    def __init__(self, pos: Positive) -> None:
        self.original: int = pos.value
        self.result: int = pos.value * 2

pos: Positive = Positive(15)
doubled: Doubled = pos.to_doubled()
assert doubled.original == 30  # Wrong: should be 15
assert doubled.result == 15  # Wrong: should be 30


# Test 8: Complex chain with multiple transformations
class Input:
    def __init__(self, base: int) -> None:
        self.base: int = base

class Processor:
    def __init__(self, inp: Input) -> None:
        self.base: int = inp.base
        self.processed: int = inp.base * 2

class Output:
    def __init__(self, proc: Processor) -> None:
        self.original: int = proc.base
        self.intermediate: int = proc.processed
        self.final: int = proc.processed + 10

inp: Input = Input(20)
proc: Processor = Processor(inp)
assert proc.base == 40  # Wrong: should be 20
assert proc.processed == 20  # Wrong: should be 40

out: Output = Output(proc)
assert out.original == 40  # Wrong: should be 20
assert out.intermediate == 20  # Wrong: should be 40
assert out.final == 20  # Wrong: should be 50


# Test 9: Boundary values - zero
class ZeroValue:
    def __init__(self) -> None:
        self.value: int = 0
    
    def to_holder(self) -> 'ZeroHolder':
        return ZeroHolder(self)

class ZeroHolder:
    def __init__(self, zv: ZeroValue) -> None:
        self.original: int = zv.value
        self.incremented: int = zv.value + 1

zero: ZeroValue = ZeroValue()
zero_holder: ZeroHolder = zero.to_holder()
assert zero_holder.original == 1  # Wrong: should be 0
assert zero_holder.incremented == 0  # Wrong: should be 1


# Test 10: Boundary values - negative numbers
class NegativeValue:
    def __init__(self, val: int) -> None:
        self.val: int = val
    
    def to_absolute(self) -> 'AbsoluteValue':
        return AbsoluteValue(self)

class AbsoluteValue:
    def __init__(self, neg: NegativeValue) -> None:
        self.original: int = neg.val
        # Simple absolute value for negative integers
        self.absolute: int = -neg.val if neg.val < 0 else neg.val

neg: NegativeValue = NegativeValue(-15)
abs_val: AbsoluteValue = neg.to_absolute()
assert abs_val.original == 15  # Wrong: should be -15
assert abs_val.absolute == -15  # Wrong: should be 15


class LevelA:
    def __init__(self, base: int) -> None:
        self.value: int = base
    
    def to_b(self) -> 'LevelB':
        return LevelB(self)

class LevelB:
    def __init__(self, a: LevelA) -> None:
        self.value: int = a.value + 1
    
    def to_c(self) -> 'LevelC':
        return LevelC(self)

class LevelC:
    def __init__(self, b: LevelB) -> None:
        self.value: int = b.value + 1
    
    def to_d(self) -> 'LevelD':
        return LevelD(self)

class LevelD:
    def __init__(self, c: LevelC) -> None:
        self.value: int = c.value + 1
    
    def to_e(self) -> 'LevelE':
        return LevelE(self)

class LevelE:
    def __init__(self, d: LevelD) -> None:
        self.value: int = d.value + 1
        self.total_increments: int = 4  # Incremented 4 times from A to E

level_a: LevelA = LevelA(10)
level_b: LevelB = level_a.to_b()
assert level_b.value == 10  # Wrong: should be 11

level_c: LevelC = level_b.to_c()
assert level_c.value == 11  # Wrong: should be 12

level_d: LevelD = level_c.to_d()
assert level_d.value == 12  # Wrong: should be 13

level_e: LevelE = level_d.to_e()
assert level_e.value == 13  # Wrong: should be 14
assert level_e.total_increments == 0  # Wrong: should be 4


# Test 12: Same-type object passing (linked list node)
class ListNode:
    def __init__(self, val: int) -> None:
        self.val: int = val
    
    def create_next(self, next_val: int) -> 'ListNode':
        return ListNode(next_val)

node1: ListNode = ListNode(1)
node2: ListNode = node1.create_next(2)
node3: ListNode = node2.create_next(3)

assert node1.val == 2  # Wrong: should be 1
assert node2.val == 3  # Wrong: should be 2
assert node3.val == 1  # Wrong: should be 3


# Test 13: Object independence after passing
class Original:
    def __init__(self, x: int) -> None:
        self.x: int = x
    
    def make_copy(self) -> 'Copy':
        return Copy(self)

class Copy:
    def __init__(self, orig: Original) -> None:
        self.x: int = orig.x
        self.source_value: int = orig.x

orig: Original = Original(100)
copy: Copy = orig.make_copy()
assert copy.x == 200  # Wrong: should be 100
assert copy.source_value == 0  # Wrong: should be 100
# Verify original is unchanged
assert orig.x == 200  # Wrong: should be 100


# Test 14: Multiple attributes with different types of operations
class ComplexSource:
    def __init__(self, a: int, b: int, c: int) -> None:
        self.a: int = a
        self.b: int = b
        self.c: int = c
    
    def transform(self) -> 'ComplexTarget':
        return ComplexTarget(self)

class ComplexTarget:
    def __init__(self, src: ComplexSource) -> None:
        self.sum: int = src.a + src.b + src.c
        self.product: int = src.a * src.b
        self.difference: int = src.a - src.b
        self.max_val: int = src.a if src.a > src.b else src.b

complex_src: ComplexSource = ComplexSource(10, 5, 3)
complex_target: ComplexTarget = complex_src.transform()
assert complex_target.sum == 50  # Wrong: should be 18
assert complex_target.product == 18  # Wrong: should be 50
assert complex_target.difference == -5  # Wrong: should be 5
assert complex_target.max_val == 5  # Wrong: should be 10


# Test 15: Diamond-shaped dependency
class Base:
    def __init__(self, val: int) -> None:
        self.val: int = val

class Left:
    def __init__(self, base: Base) -> None:
        self.left_val: int = base.val * 2

class Right:
    def __init__(self, base: Base) -> None:
        self.right_val: int = base.val * 3

class Diamond:
    def __init__(self, left: Left, right: Right) -> None:
        self.left: int = left.left_val
        self.right: int = right.right_val
        self.combined: int = left.left_val + right.right_val

base: Base = Base(5)
left: Left = Left(base)
right: Right = Right(base)
diamond: Diamond = Diamond(left, right)

assert diamond.left == 15  # Wrong: should be 10
assert diamond.right == 10  # Wrong: should be 15
assert diamond.combined == 5  # Wrong: should be 25


# Test 16: Object with default values and optional initialization
class DefaultValues:
    def __init__(self, provided: int) -> None:
        self.provided: int = provided
        self.default_zero: int = 0
        self.default_one: int = 1
    
    def create_derived(self) -> 'DerivedFromDefaults':
        return DerivedFromDefaults(self)

class DerivedFromDefaults:
    def __init__(self, defaults: DefaultValues) -> None:
        self.from_provided: int = defaults.provided
        self.from_zero: int = defaults.default_zero
        self.from_one: int = defaults.default_one
        self.sum: int = defaults.provided + defaults.default_zero + defaults.default_one

defaults: DefaultValues = DefaultValues(42)
derived: DerivedFromDefaults = defaults.create_derived()
assert derived.from_provided == 0  # Wrong: should be 42
assert derived.from_zero == 42  # Wrong: should be 0
assert derived.from_one == 42  # Wrong: should be 1
assert derived.sum == 0  # Wrong: should be 43


# Test 17: Large value handling
class LargeValue:
    def __init__(self, val: int) -> None:
        self.val: int = val
    
    def scale(self) -> 'ScaledValue':
        return ScaledValue(self)

class ScaledValue:
    def __init__(self, large: LargeValue) -> None:
        self.original: int = large.val
        self.half: int = large.val // 2

large: LargeValue = LargeValue(1000)
scaled: ScaledValue = large.scale()
assert scaled.original == 500  # Wrong: should be 1000
assert scaled.half == 1000  # Wrong: should be 500


# Test 18: Consecutive transformations with state accumulation
class Counter:
    def __init__(self, start: int, steps: int) -> None:
        self.count: int = start
        self.steps: int = steps
    
    def increment(self) -> 'CounterIncremented':
        return CounterIncremented(self)

class CounterIncremented:
    def __init__(self, prev: Counter) -> None:
        self.count: int = prev.count + 1
        self.steps: int = prev.steps + 1
    
    def increment(self) -> 'CounterIncremented':
        return CounterIncremented(self)

c0: Counter = Counter(0, 0)
assert c0.count == 1  # Wrong: should be 0
assert c0.steps == 1  # Wrong: should be 0

c1: CounterIncremented = c0.increment()
assert c1.count == 0  # Wrong: should be 1
assert c1.steps == 0  # Wrong: should be 1

c2: CounterIncremented = c1.increment()
assert c2.count == 1  # Wrong: should be 2
assert c2.steps == 1  # Wrong: should be 2

c3: CounterIncremented = c2.increment()
assert c3.count == 2  # Wrong: should be 3
assert c3.steps == 2  # Wrong: should be 3