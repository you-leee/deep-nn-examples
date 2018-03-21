from . import my_stopwords, re


open_quote_regex = (re.compile(u'([«“‘])'), r' \1 ')
close_quote_regex = (re.compile(u'([»”’])'), r' \1 ')
quote_regex = (re.compile(r'(["\'])'), r' \1 ')
PUNCTUATION = [
    (re.compile(r'([:,])([^\d])'), r' \1 \2'),
    (re.compile(r'([:,])$'), r' \1 '),
    (re.compile(r'\.\.\.'), r' ... '),
    (re.compile(r'[;@#$%&]'), r' \g<0> '),
    (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),  # final period.
    (re.compile(r'[?!]'), r' \g<0> '),
]
PARENS_BRACKETS = (re.compile(r'[\]\[\(\)\{\}\<\>]'), r' \g<0> ')

regexes = [open_quote_regex, close_quote_regex, quote_regex, PARENS_BRACKETS]

def word_tokenize(text, return_str=False):
    for regx, subs in regexes:
        text = regx.sub(subs, text)
    for regx, subs in PUNCTUATION:
        text = regx.sub(subs, text)

    return text if return_str else text.split()


def filter_stopwords(line):
    # Gets a line of text and tokenizes the words in it with filtering the stopwords
    # Returns the filtered, tokenized list of words

    line = re.sub('\t', " ", line.strip())
    words = [w for w in word_tokenize(line) if w.lower() not in my_stopwords]

    return words
