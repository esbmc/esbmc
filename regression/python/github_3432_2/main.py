from typing import BinaryIO


class API:

    def __init__(self):
        self.upload_count: int = 0
        self.last_upload_size: int = 0

    def upload(self, file: BinaryIO) -> None:
        # Duck typing - accepts any object
        # Simple operation that works with any type
        self.upload_count += 1

    def get_upload_count(self) -> int:
        return self.upload_count


api = API()
assert api.upload_count == 0

api.upload({"data": "not a file"})
assert api.upload_count == 1

api.upload({"data": "second upload"})
assert api.upload_count == 2

api.upload({"other_key": "value"})
assert api.upload_count == 3

api.upload({})
assert api.upload_count == 4

count = api.get_upload_count()
assert count == 4
