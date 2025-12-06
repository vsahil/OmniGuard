# ciphers we used

chinese_alphabet = ["甲","乙","丙","丁","戊","己","庚","辛","壬","癸","子","丑","寅","卯","辰","巳","午","未","申","酉","戌","亥","天","地","人","黄"]
english_alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
chinese_alphabet = ["e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z", "a","b","c","d"]

# for i in range(26):
#     print(english_alphabet[i], "->", chinese_alphabet[i])

class SelfDefineCipher():

    def encode(self, s):
        s = s.lower()

        ans = ""
        for letter in s:
            try:
                ans += chinese_alphabet[ord(letter.lower()) - 96-1]
            except:
                ans += letter
        return ans

    def decode(self, s):
        ans = ""
        for letter in s:
            try:
                position = chinese_alphabet.index(letter)
                ans += english_alphabet[position]
            except:
                ans += letter
        return ans


class CaesarExpert():

    def __init__(self, shift):
        self.shift = shift

    def encode(self, s):
        ans = ''
        for p in s:
            if 'a' <= p <= 'z':
                ans += chr(ord('a') + (ord(p) - ord('a') + self.shift) % 26)
            elif 'A' <= p <= 'Z':
                ans += chr(ord('A') + (ord(p) - ord('A') + self.shift) % 26)
            else:
                ans += p

        return ans

    def decode(self, s):
        ans = ''
        for p in s:
            if 'a' <= p <= 'z':
                ans += chr(ord('a') + (ord(p) - ord('a') - self.shift) % 26)
            elif 'A' <= p <= 'Z':
                ans += chr(ord('A') + (ord(p) - ord('A') - self.shift) % 26)
            else:
                ans += p
        return ans


class UnicodeExpert():

    def encode(self, s):
        ans = ''

        lines = s.split("\n")
        for line in lines:
            for c in line:
                byte_s = str(c.encode("unicode_escape"))
                if len(byte_s) > 8:
                    ans += byte_s[3:-1]
                else:
                    ans += byte_s[-2]
            ans += "\n"
        return ans

    def decode(self, s):
        ans = bytes(s, encoding="utf8").decode("unicode_escape")
        return ans


class BaseExpert():

    def encode(self, s):
        return s

    def decode(self, s):
        return s


class UTF8Expert():

    def encode(self, s):
        ans = ''

        lines = s.split("\n")
        for line in lines:
            for c in line:
                byte_s = str(c.encode("utf8"))
                if len(byte_s) > 8:
                    ans += byte_s[2:-1]
                else:
                    ans += byte_s[-2]
            ans += "\n"
        return ans

    def decode(self, s):
        ans = b''
        while len(s):
            if s.startswith("\\x"):
                ans += bytes.fromhex(s[2:4])
                s = s[4:]
            else:
                ans += bytes(s[0], encoding="utf8")
                s = s[1:]

        ans = ans.decode("utf8")
        return ans


class AsciiExpert():

    def encode(self, s):
        ans = ''

        lines = s.split("\n")
        for line in lines:
            for c in line:
                try:
                    ans += str(ord(c)) + " "
                except:
                    ans += c
            ans += "\n"
        return ans

    def decode(self, s):
        ans = ""
        lines = s.split("\n")
        for line in lines:
            cs = line.split()
            for c in cs:
                try:
                    ans += chr(int(c))
                except:
                    ans += c
        return ans


class GBKExpert():

    def encode(self, s):
        ans = ''

        lines = s.split("\n")
        for line in lines:
            for c in line:
                byte_s = str(c.encode("GBK"))
                if len(byte_s) > 8:
                    ans += byte_s[2:-1]
                else:
                    ans += byte_s[-2]
            ans += "\n"
        return ans

    def decode(self, s):
        ans = b''
        while len(s):
            if s.startswith("\\x"):
                ans += bytes.fromhex(s[2:4])
                s = s[4:]
            else:
                ans += bytes(s[0], encoding="GBK")
                s = s[1:]

        ans = ans.decode("GBK")
        return ans


class MorseExpert():

    def encode(self, s):
        s = s.upper()
        MORSE_CODE_DICT = {'A': '.-', 'B': '-...',
                           'C': '-.-.', 'D': '-..', 'E': '.',
                           'F': '..-.', 'G': '--.', 'H': '....',
                           'I': '..', 'J': '.---', 'K': '-.-',
                           'L': '.-..', 'M': '--', 'N': '-.',
                           'O': '---', 'P': '.--.', 'Q': '--.-',
                           'R': '.-.', 'S': '...', 'T': '-',
                           'U': '..-', 'V': '...-', 'W': '.--',
                           'X': '-..-', 'Y': '-.--', 'Z': '--..',
                           '1': '.----', '2': '..---', '3': '...--',
                           '4': '....-', '5': '.....', '6': '-....',
                           '7': '--...', '8': '---..', '9': '----.',
                           '0': '-----', ', ': '--..--', '.': '.-.-.-',
                           '?': '..--..', '/': '-..-.', '-': '-....-',
                           '(': '-.--.', ')': '-.--.-'}
        cipher = ''
        lines = s.split("\n")
        for line in lines:
            for letter in line:
                try:
                    if letter != ' ':
                        cipher += MORSE_CODE_DICT[letter] + ' '
                    else:
                        cipher += ' '
                except:
                    cipher += letter + ' '
            cipher += "\n"
        return cipher

    def decode(self, s):
        MORSE_CODE_DICT = {'A': '.-', 'B': '-...',
                           'C': '-.-.', 'D': '-..', 'E': '.',
                           'F': '..-.', 'G': '--.', 'H': '....',
                           'I': '..', 'J': '.---', 'K': '-.-',
                           'L': '.-..', 'M': '--', 'N': '-.',
                           'O': '---', 'P': '.--.', 'Q': '--.-',
                           'R': '.-.', 'S': '...', 'T': '-',
                           'U': '..-', 'V': '...-', 'W': '.--',
                           'X': '-..-', 'Y': '-.--', 'Z': '--..',
                           '1': '.----', '2': '..---', '3': '...--',
                           '4': '....-', '5': '.....', '6': '-....',
                           '7': '--...', '8': '---..', '9': '----.',
                           '0': '-----', ', ': '--..--', '.': '.-.-.-',
                           '?': '..--..', '/': '-..-.', '-': '-....-',
                           '(': '-.--.', ')': '-.--.-'}
        decipher = ''
        citext = ''
        lines = s.split("\n")
        for line in lines:
            for letter in line:
                while True and len(letter):
                    if letter[0] not in ['-', '.', ' ']:
                        decipher += letter[0]
                        letter = letter[1:]
                    else:
                        break
                try:
                    if (letter != ' '):
                        i = 0
                        citext += letter
                    else:
                        i += 1
                        if i == 2:
                            decipher += ' '
                        else:
                            decipher += list(MORSE_CODE_DICT.keys())[list(MORSE_CODE_DICT.values()).index(citext)]
                            citext = ''
                except:
                    decipher += letter
            decipher += '\n'
        return decipher


class AtbashExpert():
    def encode(self, text):
        ans = ''
        N = ord('z') + ord('a')
        for s in text:
            try:
                if s.isalpha():
                    ans += chr(N - ord(s))
                else:
                    ans += s
            except:
                ans += s
        return ans

    def decode(self, text):
        ans = ''
        N = ord('z') + ord('a')
        for s in text:
            try:
                if s.isalpha():
                    ans += chr(N - ord(s))
                else:
                    ans += s
            except:
                ans += s
        return ans


class AtbashCipher():
    def encode(self, s):
        result = ''
        for char in s:
            if 'a' <= char <= 'z':  # Check if the character is a lowercase letter
                result += chr(219 - ord(char))  # Transform lowercase letters
            elif 'A' <= char <= 'Z':  # Check if the character is an uppercase letter
                result += chr(155 - ord(char))  # Transform uppercase letters
            else:
                result += char  # Leave non-alphabet characters unchanged
        return result

    def decode(self, s):
        return self.encode(s)  # Atbash is its own inverse since the transformation is symmetrical
    

class VigenereCipher():
    def __init__(self, keyword):
        self.keyword = keyword
    
    def encode(self, s):
        key = self.keyword * (len(s) // len(self.keyword) + 1)
        encoded = ''
        for p, k in zip(s, key):
            if p.isalpha():
                offset = ord('A') if p.isupper() else ord('a')
                encoded += chr((ord(p) - offset + ord(k.lower()) - ord('a')) % 26 + offset)
            else:
                encoded += p
        return encoded

    def decode(self, s):
        key = self.keyword * (len(s) // len(self.keyword) + 1)
        decoded = ''
        for p, k in zip(s, key):
            if p.isalpha():
                offset = ord('A') if p.isupper() else ord('a')
                decoded += chr((ord(p) - offset - (ord(k.lower()) - ord('a'))) % 26 + offset)
            else:
                decoded += p
        return decoded


class HexadecimalExpert():
    def encode(self, s):
        return ' '.join(hex(ord(c))[2:] for c in s)

    def decode(self, s):
        return ''.join(chr(int(c, 16)) for c in s.split())


import base64
class Base64Expert():
    def encode(self, s):
        return base64.b64encode(s.encode()).decode()

    def decode(self, s):
        return base64.b64decode(s.encode()).decode()


class LeetSpeak():
    def encode(self, s):
        replacements = {
            'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7', 'l': '|', 'b': '8'
        }
        return ''.join(replacements.get(c.lower(), c) for c in s)

    def decode(self, s):
        reverse_replacements = {
            '4': 'a', '3': 'e', '1': 'i', '0': 'o', '5': 's', '7': 't', '|': 'l', '8': 'b'
        }
        return ''.join(reverse_replacements.get(c, c) for c in s)


class KeyboardShift():
    def encode(self, s):
        # Example: shift right for QWERTY layout
        keyboard = "qwertyuiopasdfghjkl;zxcvbnm,./"
        shifted = "wertyuiopasdfghjkl;zxcvbnm,./q"
        trans_table = str.maketrans(keyboard, shifted)
        return s.translate(trans_table)

    def decode(self, s):
        # Inverse of the encode example
        keyboard = "qwertyuiopasdfghjkl;zxcvbnm,./"
        shifted = "wertyuiopasdfghjkl;zxcvbnm,./q"
        trans_table = str.maketrans(shifted, keyboard)
        return s.translate(trans_table)


class VowelRemoval():
    def encode(self, s):
        vowels = 'aeiouAEIOU'
        return ''.join(c for c in s if c not in vowels)

    def decode(self, s):
        # Decoding is not typically possible as original vowels are lost
        return s        # Return original string


class AlphanumericCode():
    def encode(self, s):
        return ' '.join(str(ord(c)) if c.isalpha() else c for c in s)

    def decode(self, s):
        return ''.join(chr(int(c)) if c.isdigit() else c for c in s.split())


class ReverseText():
    def encode(self, s):
        return s[::-1]

    def decode(self, s):
        return s[::-1]  # Reversing again returns to original


class BinaryCode():
    def encode(self, s):
        return ' '.join(format(ord(c), 'b') for c in s)

    def decode(self, s):
        return ''.join(chr(int(c, 2)) for c in s.split())


class PigpenCipher():
    def __init__(self):
        # Define unique grids for Pigpen Cipher
        # Two tic-tac-toe grids without reuse
        self.grid = {
            'A': '┌', 'B': '┐', 'C': '└', 'D': '┘', 'E': '━', 'F': '┃', 'G': '┏', 'H': '┓', 'I': '┛',
            'J': '┗', 'K': '┳', 'L': '┻', 'M': '┣', 'N': '┫', 'O': '╋', 'P': '╸', 'Q': '╺', 'R': '╹',
            'S': '╻', 'T': '╴', 'U': '╵', 'V': '╷', 'W': '╱', 'X': '╲', 'Y': '╳', 'Z': '═'
        }

    def encode(self, text):
        encoded_text = ''
        for char in text.upper():
            if char in self.grid:
                encoded_text += self.grid[char] + " "
            else:
                encoded_text += char + " "  # Non-alphabet characters are left as is
        return encoded_text.strip()

    def decode(self, text):
        # Reverse the grid for decoding
        reverse_grid = {v: k for k, v in self.grid.items()}
        decoded_text = ''
        symbols = text.split()
        for symbol in symbols:
            if symbol in reverse_grid:
                decoded_text += reverse_grid[symbol]
            else:
                decoded_text += symbol  # Non-symbol characters are left as is
        return decoded_text


encode_expert_dict = {
    "unchange": BaseExpert(),
    "baseline": BaseExpert(),
    "caesar": CaesarExpert(shift=3),
    "caesar1": CaesarExpert(shift=1),
    "caesar2": CaesarExpert(shift=2),
    "caesar4": CaesarExpert(shift=4),
    "caesar5": CaesarExpert(shift=5),
    "caesar6": CaesarExpert(shift=6),
    "caesar7": CaesarExpert(shift=7),
    "caesar8": CaesarExpert(shift=8),
    "caesar9": CaesarExpert(shift=9),
    "caesarneg1": CaesarExpert(shift=-1),
    "caesarneg2": CaesarExpert(shift=-2),
    "caesarneg3": CaesarExpert(shift=-3),
    "caesarneg4": CaesarExpert(shift=-4),
    "caesarneg5": CaesarExpert(shift=-5),
    "caesarneg6": CaesarExpert(shift=-6),
    "caesarneg7": CaesarExpert(shift=-7),
    "caesarneg8": CaesarExpert(shift=-8),
    "caesarneg9": CaesarExpert(shift=-9),
    "unicode": UnicodeExpert(),
    "morse": MorseExpert(),
    "utf": UTF8Expert(),
    "ascii": AsciiExpert(),
    "gbk": GBKExpert(),
    "selfdefine": SelfDefineCipher(),
    "atbash": AtbashCipher(),
    "vigenere": VigenereCipher("nooneknows"),
    "hexadecimal": HexadecimalExpert(),
    "base64": Base64Expert(),
    "leet": LeetSpeak(),
    "keyboard": KeyboardShift(),
    "vowel": VowelRemoval(),
    "alphanumeric": AlphanumericCode(),
    "reverse": ReverseText(),
    "binary": BinaryCode(),
    "pigpen": PigpenCipher()
}

