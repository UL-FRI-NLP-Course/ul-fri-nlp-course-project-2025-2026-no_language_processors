# Text Processing Comparison Report

Fill in the analysis prompts under each sample after running the script.

### Sample 1
**Source Text:** Wait... did Dr. J. Smith (U.C. Berkeley) really say 'NLP is easy' at 3:30 p.m., or was it sarcasm?!

#### Performance
- **Time Elapsed:** NLTK: 3.3225s | Stanza: 0.2018s

#### Counts
- **Tokens:** NLTK: ['Wait', '...', 'did', 'Dr.', 'J.', 'Smith', '(', 'U.C', '.', 'Berkeley', ')', 'really', 'say', "'NLP", 'is', 'easy', "'", 'at', '3:30', 'p.m.', ',', 'or', 'was', 'it', 'sarcasm', '?', '!'] | Stanza: ['Wait', '...', 'did', 'Dr.', 'J.', 'Smith', '(', 'U.C.', 'Berkeley', ')', 'really', 'say', "'", 'NLP', 'is', 'easy', "'", 'at', '3:30', 'p.m.', ',', 'or', 'was', 'it', 'sarcasm', '?!']
- **Entities:** NLTK: (S
  (GPE Wait/NN)
  .../:
  did/VBD
  Dr./NNP
  J./NNP
  Smith/NNP
  (/(
  U.C/NNP
  ./.
  (PERSON Berkeley/NNP)
  )/)
  really/RB
  say/VB
  'NLP/NN
  is/VBZ
  easy/JJ
  '/''
  at/IN
  3:30/CD
  p.m./NN
  ,/,
  or/CC
  was/VBD
  it/PRP
  sarcasm/VB
  ?/.
  !/.) | Stanza: [('J. Smith', 'PERSON'), ('U.C. Berkeley', 'ORG'), ('3:30 p.m.', 'TIME')]

#### Previews
- **NLTK Tokens (preview):** `['Wait', '...', 'did', 'Dr.', 'J.', 'Smith', '(', 'U.C', '.', 'Berkeley', ')', 'really']`
- **Stanza Tokens (preview):** `['Wait', '...', 'did', 'Dr.', 'J.', 'Smith', '(', 'U.C.', 'Berkeley', ')', 'really', 'say']`


---

### Sample 2
**Source Text:** Email me at first.last+nlp@uni-example.edu ASAP - unless you've already sent it via https://tinyurl.com/nlp-demo.

#### Performance
- **Time Elapsed:** NLTK: 0.5733s | Stanza: 0.2192s

#### Counts
- **Tokens:** NLTK: ['Email', 'me', 'at', 'first.last+nlp', '@', 'uni-example.edu', 'ASAP', '-', 'unless', 'you', "'ve", 'already', 'sent', 'it', 'via', 'https', ':', '//tinyurl.com/nlp-demo', '.'] | Stanza: ['Email', 'me', 'at', 'first.last+nlp@uni-example.edu', 'ASAP', '-', 'unless', 'you', "'ve", 'already', 'sent', 'it', 'via', 'https://tinyurl.com/nlp-demo.']
- **Entities:** NLTK: (S
  Email/VB
  me/PRP
  at/IN
  first.last+nlp/JJ
  @/JJ
  uni-example.edu/JJ
  (ORGANIZATION ASAP/NNP)
  -/:
  unless/IN
  you/PRP
  've/VBP
  already/RB
  sent/VBN
  it/PRP
  via/IN
  https/NN
  :/:
  //tinyurl.com/nlp-demo/NN
  ./.) | Stanza: []

#### Previews
- **NLTK Tokens (preview):** `['Email', 'me', 'at', 'first.last+nlp', '@', 'uni-example.edu', 'ASAP', '-', 'unless', 'you', "'ve", 'already']`
- **Stanza Tokens (preview):** `['Email', 'me', 'at', 'first.last+nlp@uni-example.edu', 'ASAP', '-', 'unless', 'you', "'ve", 'already', 'sent', 'it']`


---

### Sample 3
**Source Text:** The startup's Q4 revenue was $1.2M-ish (not audited), yet users wrote: 'app crashes on iOS17/Android14 :('

#### Performance
- **Time Elapsed:** NLTK: 0.6045s | Stanza: 0.2460s

#### Counts
- **Tokens:** NLTK: ['The', 'startup', "'s", 'Q4', 'revenue', 'was', '$', '1.2M-ish', '(', 'not', 'audited', ')', ',', 'yet', 'users', 'wrote', ':', "'app", 'crashes', 'on', 'iOS17/Android14', ':', '(', "'"] | Stanza: ['The', 'startup', "'s", 'Q4', 'revenue', 'was', '$', '1.2', 'M', '-', 'ish', '(', 'not', 'audited', ')', ',', 'yet', 'users', 'wrote', ':', "'", 'app', 'crashes', 'on', 'iOS17/Android14', ':(', "'"]
- **Entities:** NLTK: (S
  The/DT
  startup/NN
  's/POS
  Q4/NNP
  revenue/NN
  was/VBD
  $/$
  1.2M-ish/JJ
  (/(
  not/RB
  audited/VBN
  )/)
  ,/,
  yet/CC
  users/NNS
  wrote/VBD
  :/:
  'app/CD
  crashes/NNS
  on/IN
  iOS17/Android14/NN
  :/:
  (/(
  '/'') | Stanza: [('$1.2M', 'MONEY')]

#### Previews
- **NLTK Tokens (preview):** `['The', 'startup', "'s", 'Q4', 'revenue', 'was', '$', '1.2M-ish', '(', 'not', 'audited', ')']`
- **Stanza Tokens (preview):** `['The', 'startup', "'s", 'Q4', 'revenue', 'was', '$', '1.2', 'M', '-', 'ish', '(']`


---

### Sample 4
**Source Text:** I re-read the note: "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo." Still parsing it...

#### Performance
- **Time Elapsed:** NLTK: 0.5983s | Stanza: 0.1832s

#### Counts
- **Tokens:** NLTK: ['I', 're-read', 'the', 'note', ':', '``', 'Buffalo', 'buffalo', 'Buffalo', 'buffalo', 'buffalo', 'buffalo', 'Buffalo', 'buffalo', '.', "''", 'Still', 'parsing', 'it', '...'] | Stanza: ['I', 're-read', 'the', 'note', ':', '"', 'Buffalo', 'buffalo', 'Buffalo', 'buffalo', 'buffalo', 'buffalo', 'Buffalo', 'buffalo', '.', '"', 'Still', 'parsing', 'it', '...']
- **Entities:** NLTK: (S
  I/PRP
  re-read/VBP
  the/DT
  note/NN
  :/:
  ``/``
  (PERSON Buffalo/NNP)
  buffalo/NN
  (PERSON Buffalo/NNP)
  buffalo/NN
  buffalo/NN
  buffalo/NN
  (PERSON Buffalo/NNP)
  buffalo/NN
  ./.
  ''/''
  Still/RB
  parsing/VBG
  it/PRP
  .../:) | Stanza: [('Buffalo', 'GPE'), ('Buffalo', 'GPE'), ('Buffalo', 'GPE')]

#### Previews
- **NLTK Tokens (preview):** `['I', 're-read', 'the', 'note', ':', '``', 'Buffalo', 'buffalo', 'Buffalo', 'buffalo', 'buffalo', 'buffalo']`
- **Stanza Tokens (preview):** `['I', 're-read', 'the', 'note', ':', '"', 'Buffalo', 'buffalo', 'Buffalo', 'buffalo', 'buffalo', 'buffalo']`


---
