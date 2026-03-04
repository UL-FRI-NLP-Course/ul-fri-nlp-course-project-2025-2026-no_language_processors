"""
Alternative Text Processing Assignment (Student Version)
=======================================================

Complete the TODO sections to build a full NLTK vs Stanza comparison pipeline.
This file is intentionally scaffolded for students.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
from nltk import ne_chunk, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

import stanza


nltk.download('punkt_tab') #extra line, we need this model. 
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')

TEXTS = [
    "Wait... did Dr. J. Smith (U.C. Berkeley) really say 'NLP is easy' at 3:30 p.m., or was it sarcasm?!",
    "Email me at first.last+nlp@uni-example.edu ASAP - unless you've already sent it via https://tinyurl.com/nlp-demo.",
    "The startup's Q4 revenue was $1.2M-ish (not audited), yet users wrote: 'app crashes on iOS17/Android14 :('",
    "I re-read the note: \"Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.\" Still parsing it...",
]


@dataclass
class PipelineResult:
    sentences: list[str]
    tokens: list[str]
    pos_tags: list[tuple[str, str]]
    lemmas: list[str]
    entities: list[tuple[str, str]]
    elapsed_s: float


class NLTKPipeline:
    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()

    def process(self, text: str) -> PipelineResult:
        t0 = time.perf_counter()

        # TODO: Sentence tokenization using NLTK.
        sentences = nltk.sent_tokenize(text)

        # TODO: Word tokenization using NLTK.x
        tokens = nltk.word_tokenize(text)

        # TODO: POS tagging using NLTK.
        pos_tags_result = nltk.pos_tag(tokens)

        # TODO: Lemmatize each token using WordNetLemmatizer.
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]

        # TODO: Named Entity Recognition with ne_chunk over POS tags.
        # Store entities as tuples: (entity_text, entity_label).
        tagged = nltk.pos_tag(tokens)
        entities: list[tuple[str, str]] = nltk.chunk.ne_chunk(tagged)

        return PipelineResult(
            sentences=sentences,
            tokens=tokens,
            pos_tags=pos_tags_result,
            lemmas=lemmas,
            entities=entities,
            elapsed_s=time.perf_counter() - t0,
        )


class StanzaPipeline:
    def __init__(self) -> None:
        # TODO 5.1: Initialize a Stanza pipeline for English with correct arguments
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,ner')
        
    def process(self, text: str) -> PipelineResult:
        t0 = time.perf_counter()

        # TODO: Run Stanza pipeline on text.
        doc = self.nlp(text)

        # TODO: Extract sentence texts.
        sentences = [sentence.text for sentence in doc.sentences]

        # TODO: Extract tokens (word text).
        tokens = [word.text for sentence in doc.sentences for word in sentence.words]

        # TODO: Extract POS tags as (word, tag).
        pos_tags_result = [token.upos for sentence in doc.sentences for token in sentence.words]

        # TODO: Extract lemmas.
        lemmas = [token.lemma for sentence in doc.sentences for token in sentence.words]

        # TODO: Extract named entities as (entity_text, entity_type).
        entities = [(ent.text, ent.type) for ent in doc.ents]

        return PipelineResult(
            sentences=sentences,
            tokens=tokens,
            pos_tags=pos_tags_result,
            lemmas=lemmas,
            entities=entities,
            elapsed_s=time.perf_counter() - t0,
        )


def compare_counts(nltk_res: PipelineResult, stanza_res: PipelineResult) -> dict[str, int]:
    # TODO: Return a dictionary with these keys:
    # sentences_nltk, sentences_stanza, tokens_nltk, tokens_stanza,
    # entities_nltk, entities_stanza
    return {
        "sentences_nltk": nltk_res.sentences,
        "sentences_stanza": stanza_res.sentences,
        "tokens_nltk": nltk_res.tokens,
        "tokens_stanza": stanza_res.tokens,
        "entities_nltk": nltk_res.entities,
        "entities_stanza": stanza_res.entities,
    }


def visualize_token_counts(rows: list[dict[str, str]], output_path: Path) -> None:
    # TODO: Build labels: S1, S2, ... based on number of rows.
    
    labels = ['S' + str(i) for i in range(1, len(rows) + 1)]

    # TODO: Read token counts from rows and convert to int.
    nltk_counts = [int(len(row["tokens_nltk"])) for row in rows]
    stanza_counts = [int(len(row["tokens_stanza"])) for row in rows]

    # TODO: Create a grouped bar chart (NLTK vs Stanza) and save it.
    # Requirements:


    # - X-axis: sample labels
    # - Y-axis: number of tokens
    # - title: "Token Count Comparison: NLTK vs Stanza"
    # - legend enabled
    # - save to output_path with dpi=150
    plt.figure(dpi = 150)
    plt.bar(labels, nltk_counts, label='NLTK')
    plt.bar(labels, stanza_counts, label='Stanza')
    plt.xlabel('Sample')
    plt.ylabel('Token Count')
    plt.title('Token Count Comparison: NLTK vs Stanza')
    plt.legend()
    plt.savefig(output_path)
    #pass


def write_report(rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        "# Text Processing Comparison Report",
        "",
        "Fill in the analysis prompts under each sample after running the script.",
        "",
    ]

    for i, row in enumerate(rows, start=1):
        # TODO: Add a markdown block for each sample containing:
        # - sample heading
        # - source text
        # - timings
        # - token/entity counts
        # - quick inspection previews
        # - analysis prompts
        # Tip: use lines.extend([...])
        
        # - timings
        # - token/entity counts
        # - quick inspection previews
        # - analysis prompts
        # Tip: use lines.extend([...])

        #print(f"\n{'='*20}")
        #print(f"Processing Sample {i}: {row['text'][:40]}...")
        #print(f"  > Timings: NLTK ({row['nltk_time']:.4f}s) vs Stanza ({row['stanza_time']:.4f}s)")
        #print(f"  > Token Count (Stanza): {row['tokens_stanza']}")
        #print(f"{'='*20}")

        lines.extend([
            f"### Sample {i}",
            f"**Source Text:** {row['text']}",
            "",
            "#### Performance",
            f"- **Time Elapsed:** NLTK: {row['nltk_time']:.4f}s | Stanza: {row['stanza_time']:.4f}s",
            "",
            "#### Counts",
            f"- **Tokens:** NLTK: {row['tokens_nltk']} | Stanza: {row['tokens_stanza']}",
            f"- **Entities:** NLTK: {row['entities_nltk']} | Stanza: {row['entities_stanza']}",
            "",
            "#### Previews",
            f"- **NLTK Tokens (preview):** `{row['nltk_tokens_preview']}`",
            f"- **Stanza Tokens (preview):** `{row['stanza_tokens_preview']}`",
            "",
            "",
            "---", 
            ""
        ])


    output_path.write_text("\n".join(lines), encoding="utf-8")


def ensure_nltk_resources() -> None:
    required = [
        "punkt",
        "averaged_perceptron_tagger",
        "wordnet",
        "maxent_ne_chunker",
        "words",
    ]
    for item in required:
        nltk.download(item, quiet=True)


def run() -> None:
    ensure_nltk_resources()
    stanza.download("en", verbose=False)

    nltk_pipe = NLTKPipeline()
    stanza_pipe = StanzaPipeline()

    report_rows: list[dict[str, str]] = []

    for text in TEXTS:
        nltk_res = nltk_pipe.process(text)
        stanza_res = stanza_pipe.process(text)
        counts = compare_counts(nltk_res, stanza_res)

        # TODO: Append a dictionary to report_rows containing:

        # text, nltk_time, stanza_time,
        # tokens_nltk, tokens_stanza,
        # entities_nltk, entities_stanza,
        # nltk_tokens_preview, stanza_tokens_preview,
        # nltk_pos_preview, stanza_pos_preview,
        # nltk_entities, stanza_entities
        # Use previews: first 12 tokens and first 8 POS tags.

        results = {
            "text": text,
            "nltk_time": nltk_res.elapsed_s,
            "stanza_time": stanza_res.elapsed_s,
            "tokens_nltk": counts["tokens_nltk"],
            "tokens_stanza": counts["tokens_stanza"],
            "entities_nltk": counts["entities_nltk"],
            "entities_stanza": counts["entities_stanza"],
            "nltk_tokens_preview": nltk_res.tokens[:12],
            "stanza_tokens_preview": stanza_res.tokens[:12],
            "nltk_pos_preview": nltk_res.pos_tags[:8],
            "stanza_pos_preview": stanza_res.pos_tags[:8],
        }

        report_rows.append(results)

        #pass

    report_path = Path("rext_processing_alternative_report_guillem.md")
    plot_path = Path("token_count_comparison_guillem.png")

    # TODO: Call write_report and visualize_token_counts using report_rows.
    write_report(report_rows, report_path)
    visualize_token_counts(report_rows, plot_path)

    print("Student assignment executed.")
    print("Complete all TODO sections to generate full output files.")

if __name__ == "__main__":
    run()
