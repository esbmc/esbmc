from typing import BinaryIO

class API:
    def upload(self, file: BinaryIO) -> None:
        pass

api = API()
api.upload({"data": "not a file"})
