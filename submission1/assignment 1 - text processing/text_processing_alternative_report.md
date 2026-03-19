# Text Processing Comparison Report

Fill in the analysis prompts under each sample after running the script.

## Sample 1

**Source Text:**  
Wait... did Dr. J. Smith (U.C. Berkeley) really say 'NLP is easy' at 3:30 p.m., or was it sarcasm?!

**Timings:**
- NLTK: 2.3762s
- Stanza: 0.4531s

**Token Counts:**
- NLTK: 27
- Stanza: 26

**Entity Counts:**
- NLTK: 2
- Stanza: 3

**Token Preview (first 12):**
- NLTK: ['Wait', '...', 'did', 'Dr.', 'J.', 'Smith', '(', 'U.C', '.', 'Berkeley', ')', 'really']
- Stanza: ['Wait', '...', 'did', 'Dr.', 'J.', 'Smith', '(', 'U.C.', 'Berkeley', ')', 'really', 'say']

**POS Preview (first 8):**
- NLTK: [('Wait', 'NN'), ('...', ':'), ('did', 'VBD'), ('Dr.', 'NNP'), ('J.', 'NNP'), ('Smith', 'NNP'), ('(', '('), ('U.C', 'NNP')]
- Stanza: [('Wait', 'VERB'), ('...', 'PUNCT'), ('did', 'AUX'), ('Dr.', 'PROPN'), ('J.', 'PROPN'), ('Smith', 'PROPN'), ('(', 'PUNCT'), ('U.C.', 'PROPN')]

**Named Entities:**
- NLTK: [('Wait', 'GPE'), ('Berkeley', 'PERSON')]
- Stanza: [('J. Smith', 'PERSON'), ('U.C. Berkeley', 'ORG'), ('3:30 p.m.', 'TIME')]

**Analysis Prompts:**
- What differences do you notice in tokenization between NLTK and Stanza?
- How do the POS tags differ? Which seems more accurate?
- Compare the named entities identified by each tool.
- Which tool performed better on this specific text? Why?

## Sample 2

**Source Text:**  
Email me at first.last+nlp@uni-example.edu ASAP - unless you've already sent it via https://tinyurl.com/nlp-demo.

**Timings:**
- NLTK: 0.4263s
- Stanza: 0.4901s

**Token Counts:**
- NLTK: 19
- Stanza: 14

**Entity Counts:**
- NLTK: 1
- Stanza: 0

**Token Preview (first 12):**
- NLTK: ['Email', 'me', 'at', 'first.last+nlp', '@', 'uni-example.edu', 'ASAP', '-', 'unless', 'you', "'ve", 'already']
- Stanza: ['Email', 'me', 'at', 'first.last+nlp@uni-example.edu', 'ASAP', '-', 'unless', 'you', "'ve", 'already', 'sent', 'it']

**POS Preview (first 8):**
- NLTK: [('Email', 'VB'), ('me', 'PRP'), ('at', 'IN'), ('first.last+nlp', 'JJ'), ('@', 'JJ'), ('uni-example.edu', 'JJ'), ('ASAP', 'NNP'), ('-', ':')]
- Stanza: [('Email', 'VERB'), ('me', 'PRON'), ('at', 'ADP'), ('first.last+nlp@uni-example.edu', 'PROPN'), ('ASAP', 'ADV'), ('-', 'PUNCT'), ('unless', 'SCONJ'), ('you', 'PRON')]

**Named Entities:**
- NLTK: [('ASAP', 'ORGANIZATION')]
- Stanza: []

**Analysis Prompts:**
- What differences do you notice in tokenization between NLTK and Stanza?
- How do the POS tags differ? Which seems more accurate?
- Compare the named entities identified by each tool.
- Which tool performed better on this specific text? Why?

## Sample 3

**Source Text:**  
The startup's Q4 revenue was $1.2M-ish (not audited), yet users wrote: 'app crashes on iOS17/Android14 :('

**Timings:**
- NLTK: 0.4353s
- Stanza: 0.4994s

**Token Counts:**
- NLTK: 24
- Stanza: 27

**Entity Counts:**
- NLTK: 0
- Stanza: 1

**Token Preview (first 12):**
- NLTK: ['The', 'startup', "'s", 'Q4', 'revenue', 'was', '$', '1.2M-ish', '(', 'not', 'audited', ')']
- Stanza: ['The', 'startup', "'s", 'Q4', 'revenue', 'was', '$', '1.2', 'M', '-', 'ish', '(']

**POS Preview (first 8):**
- NLTK: [('The', 'DT'), ('startup', 'NN'), ("'s", 'POS'), ('Q4', 'NNP'), ('revenue', 'NN'), ('was', 'VBD'), ('$', '$'), ('1.2M-ish', 'JJ')]
- Stanza: [('The', 'DET'), ('startup', 'NOUN'), ("'s", 'PART'), ('Q4', 'PROPN'), ('revenue', 'NOUN'), ('was', 'AUX'), ('$', 'SYM'), ('1.2', 'NUM')]

**Named Entities:**
- NLTK: []
- Stanza: [('$1.2M', 'MONEY')]

**Analysis Prompts:**
- What differences do you notice in tokenization between NLTK and Stanza?
- How do the POS tags differ? Which seems more accurate?
- Compare the named entities identified by each tool.
- Which tool performed better on this specific text? Why?

## Sample 4

**Source Text:**  
I re-read the note: "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo." Still parsing it...

**Timings:**
- NLTK: 0.4330s
- Stanza: 0.4048s

**Token Counts:**
- NLTK: 20
- Stanza: 20

**Entity Counts:**
- NLTK: 3
- Stanza: 3

**Token Preview (first 12):**
- NLTK: ['I', 're-read', 'the', 'note', ':', '``', 'Buffalo', 'buffalo', 'Buffalo', 'buffalo', 'buffalo', 'buffalo']
- Stanza: ['I', 're-read', 'the', 'note', ':', '"', 'Buffalo', 'buffalo', 'Buffalo', 'buffalo', 'buffalo', 'buffalo']

**POS Preview (first 8):**
- NLTK: [('I', 'PRP'), ('re-read', 'VBP'), ('the', 'DT'), ('note', 'NN'), (':', ':'), ('``', '``'), ('Buffalo', 'NNP'), ('buffalo', 'NN')]
- Stanza: [('I', 'PRON'), ('re-read', 'VERB'), ('the', 'DET'), ('note', 'NOUN'), (':', 'PUNCT'), ('"', 'PUNCT'), ('Buffalo', 'PROPN'), ('buffalo', 'PROPN')]

**Named Entities:**
- NLTK: [('Buffalo', 'PERSON'), ('Buffalo', 'PERSON'), ('Buffalo', 'PERSON')]
- Stanza: [('Buffalo', 'GPE'), ('Buffalo', 'GPE'), ('Buffalo', 'GPE')]

**Analysis Prompts:**
- What differences do you notice in tokenization between NLTK and Stanza?
- How do the POS tags differ? Which seems more accurate?
- Compare the named entities identified by each tool.
- Which tool performed better on this specific text? Why?
