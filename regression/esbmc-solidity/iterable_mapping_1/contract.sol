// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.8.8;

// Iterable Mappings — from Solidity docs "Types / Mapping Types".
// Stress-tests: free-standing structs, user-defined value types (wrap/unwrap),
// libraries with storage params, using-for, delete on mapping element,
// tuple destructuring with omitted slot, while-loop iterator.

struct IndexValue { uint keyIndex; uint value; }
struct KeyFlag { uint key; bool deleted; }

struct itmap {
    mapping(uint => IndexValue) data;
    KeyFlag[] keys;
    uint size;
}

type Iterator is uint;

library IterableMapping {
    function insert(itmap storage self, uint key, uint value) internal returns (bool replaced) {
        uint keyIndex = self.data[key].keyIndex;
        self.data[key].value = value;
        if (keyIndex > 0)
            return true;
        else {
            keyIndex = self.keys.length;
            self.keys.push();
            self.data[key].keyIndex = keyIndex + 1;
            self.keys[keyIndex].key = key;
            self.size++;
            return false;
        }
    }

    function remove(itmap storage self, uint key) internal returns (bool success) {
        uint keyIndex = self.data[key].keyIndex;
        if (keyIndex == 0)
            return false;
        delete self.data[key];
        self.keys[keyIndex - 1].deleted = true;
        self.size--;
        return true;
    }

    function contains(itmap storage self, uint key) internal view returns (bool) {
        return self.data[key].keyIndex > 0;
    }

    function iterateStart(itmap storage self) internal view returns (Iterator) {
        return iteratorSkipDeleted(self, 0);
    }

    function iterateValid(itmap storage self, Iterator iterator) internal view returns (bool) {
        return Iterator.unwrap(iterator) < self.keys.length;
    }

    function iterateNext(itmap storage self, Iterator iterator) internal view returns (Iterator) {
        return iteratorSkipDeleted(self, Iterator.unwrap(iterator) + 1);
    }

    function iterateGet(itmap storage self, Iterator iterator) internal view returns (uint key, uint value) {
        uint keyIndex = Iterator.unwrap(iterator);
        key = self.keys[keyIndex].key;
        value = self.data[key].value;
    }

    function iteratorSkipDeleted(itmap storage self, uint keyIndex) private view returns (Iterator) {
        while (keyIndex < self.keys.length && self.keys[keyIndex].deleted)
            keyIndex++;
        return Iterator.wrap(keyIndex);
    }
}

contract User {
    itmap data;
    using IterableMapping for itmap;

    function insert(uint k, uint v) public returns (uint size) {
        data.insert(k, v);
        return data.size;
    }

    function sum() public view returns (uint s) {
        for (
            Iterator i = data.iterateStart();
            data.iterateValid(i);
            i = data.iterateNext(i)
        ) {
            (, uint value) = data.iterateGet(i);
            s += value;
        }
    }

    function test_insert_sum() public {
        uint s;
        s = insert(10, 100);
        assert(s == 1);
        s = insert(20, 200);
        assert(s == 2);
        assert(sum() == 300);

        // overwrite existing key
        insert(10, 150);
        assert(sum() == 350);
    }

    function test_remove() public {
        insert(10, 100);
        insert(20, 200);
        insert(30, 300);
        assert(data.size == 3);

        bool ok = data.remove(20);
        assert(ok);
        assert(data.size == 2);
        assert(!data.contains(20));
        assert(data.contains(10));
        assert(data.contains(30));
        assert(sum() == 400);
    }
}
