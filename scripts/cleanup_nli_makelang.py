import sys

LANG_MAP = {
        'ARA': 'AR',
        'CHI': 'ZH',
        'FRE': 'FR',
        'GER': 'DE',
        'HIN': 'HI',
        'ITA': 'IT',
        'JPN': 'JA',
        'KOR': 'KO',
        'SPA': 'ES',
        'TEL': 'TE',
        'TUR': 'TR'
        }

langs = list(map(str.strip, open(sys.argv[1])))
ns = list(map(int, open(sys.argv[2])))
for lang, n in zip(langs, ns):
    lang = LANG_MAP[lang]
    for i in range(n):
        print(lang)

