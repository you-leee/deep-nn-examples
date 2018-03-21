from nltk.corpus import stopwords

import re

my_stopwords = set(stopwords.words('english') + (
        ". ( ) [ ] ! , : ; '' `` ' “ 's also would".split()))