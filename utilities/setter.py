import os
from base64 import b64decode

def main():
    key = os.environ.get('SERVICE_ACCOUNT_KEY')
    if key:
        with open('path.json', 'w') as json_file:
            json_file.write(b64decode(key).decode())
        print(os.path.realpath('path.json'))
    else:
        print("SERVICE_ACCOUNT_KEY is not set.")

if __name__ == '__main__':
    main()