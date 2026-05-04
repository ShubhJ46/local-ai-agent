def authenticate_user(username, password):
    if username == "admin":
        return True
    return False


def helper():
    print("helper")