from commons.text_utils.filter_stopwords import filter_stopwords


def readDoc(path, name, extension, tokenize=False):
    fp = open(path + name + "." + extension, "r", encoding="utf8")
    docs = []
    tokenized_doc = []
    max_doc_len = 0

    line = fp.readline()
    while line:
        words = line.split()
        if (len(words) > 1):
            if (tokenize):
                words = filter_stopwords(line)
                tokenized_doc.append(words)
            if (len(words) > max_doc_len):
                max_doc_len = len(words)
            docs.append(' '.join(words))
        line = fp.readline()

    fp.close()

    return docs, max_doc_len, tokenized_doc


def load_apple():
    company_docs, c_max_len, company_tokens = readDoc("../datasets/", "appleinc", "txt", True)
    fruit_docs, f_max_len, fruit_tokens = readDoc("../datasets/", "applefruit", "txt", True)

    max_doc_len = max(c_max_len, f_max_len)

    return company_docs, fruit_docs, company_tokens, fruit_tokens, max_doc_len
