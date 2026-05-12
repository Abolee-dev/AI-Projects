import time

class TokenManager:
    def __init__(self):
        self.token = None
        self.expiry_epoch = 0

    def get_token(self) -> str:
        if self.token and time.time() < self.expiry_epoch - 60:
            return self.token

        # Replace with internal token call.
        self.token = "mock-token"
        self.expiry_epoch = time.time() + 1800
        return self.token
