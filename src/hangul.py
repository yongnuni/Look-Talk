from jamo import j2h

# ── 한글 자모 ─────────────────────────────────────────────────

CHOSUNG = [
    'ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ',
    'ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ',
    'ㅋ','ㅌ','ㅍ','ㅎ'
]

JUNGSUNG = [
    'ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ',
    'ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ',
    'ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ'
]

JONGSUNG = [
    '',
    'ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ',
    'ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ',
    'ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ',
    'ㅊ','ㅋ','ㅌ','ㅍ','ㅎ'
]

double_consonants = {
    'ㄱ':'ㄲ',
    'ㄷ':'ㄸ',
    'ㅂ':'ㅃ',
    'ㅅ':'ㅆ',
    'ㅈ':'ㅉ'
}

compound_vowels = {
    ('ㅗ','ㅏ'):'ㅘ',
    ('ㅗ','ㅐ'):'ㅙ',
    ('ㅗ','ㅣ'):'ㅚ',
    ('ㅜ','ㅓ'):'ㅝ',
    ('ㅜ','ㅔ'):'ㅞ',
    ('ㅜ','ㅣ'):'ㅟ',
    ('ㅡ','ㅣ'):'ㅢ'
}

double_jongsung = {
    ('ㄱ','ㅅ'):'ㄳ',
    ('ㄴ','ㅈ'):'ㄵ',
    ('ㄴ','ㅎ'):'ㄶ',
    ('ㄹ','ㄱ'):'ㄺ',
    ('ㄹ','ㅁ'):'ㄻ',
    ('ㄹ','ㅂ'):'ㄼ',
    ('ㄹ','ㅅ'):'ㄽ',
    ('ㄹ','ㅌ'):'ㄾ',
    ('ㄹ','ㅍ'):'ㄿ',
    ('ㄹ','ㅎ'):'ㅀ',
    ('ㅂ','ㅅ'):'ㅄ'
}

jamo_buffer = ['', '', '']
finalText = ""

def is_choseong(c):
    return c in CHOSUNG

def is_jungseong(c):
    return c in JUNGSUNG

def is_jongseong(c):
    return c in JONGSUNG and c != ''

def compose_jamo_buffer():
    if jamo_buffer[0] and jamo_buffer[1]:
        try:
            return j2h(*jamo_buffer)
        except:
            return ''.join(jamo_buffer)

    elif jamo_buffer[1]:
        return jamo_buffer[1]

    elif jamo_buffer[0]:
        return jamo_buffer[0]

    return ''

def flush_buffer():
    global finalText

    if jamo_buffer[0] != '':
        finalText += compose_jamo_buffer()

    jamo_buffer[:] = ['', '', '']

def add_jamo(j):
    global finalText

    if jamo_buffer[1] and jamo_buffer[2] == '' and is_jungseong(j):
        combined = compound_vowels.get((jamo_buffer[1], j))
        if combined:
            jamo_buffer[1] = combined
            return

    if jamo_buffer == ['', '', '']:

        if is_choseong(j):
            jamo_buffer[0] = j

        elif is_jungseong(j):
            jamo_buffer[1] = j

        else:
            jamo_buffer[0] = j

    elif jamo_buffer[1] == '':

        if is_jungseong(j):
            jamo_buffer[1] = j

        else:
            flush_buffer()
            add_jamo(j)

    elif jamo_buffer[2] == '':

        if is_jongseong(j):
            jamo_buffer[2] = j

        elif is_jungseong(j):
            flush_buffer()
            finalText += j

        elif is_choseong(j):
            flush_buffer()
            jamo_buffer[0] = j

        else:
            flush_buffer()
            jamo_buffer[0] = j

    else:

        if is_jongseong(j):

            combo = double_jongsung.get((jamo_buffer[2], j))

            if combo:
                jamo_buffer[2] = combo

            else:
                flush_buffer()
                jamo_buffer[0] = j

        elif is_jungseong(j):

            jong = jamo_buffer[2]

            if jong in double_jongsung.values():

                for k, v in double_jongsung.items():

                    if v == jong:
                        first_part, second_part = k
                        break

                else:
                    first_part, second_part = jong, ''

                finalText += j2h(
                    jamo_buffer[0],
                    jamo_buffer[1],
                    first_part
                )

                jamo_buffer[:] = [second_part, j, '']

            else:

                prev_jong = jamo_buffer[2]

                jamo_buffer[2] = ''

                finalText += compose_jamo_buffer()

                jamo_buffer[:] = [prev_jong, j, '']

        elif is_choseong(j):

            flush_buffer()

            jamo_buffer[0] = j

        else:

            flush_buffer()

            jamo_buffer[0] = j