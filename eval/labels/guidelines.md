# Labelling guidelines

The same definitions go into the prompts of the judge, so that both measure the same thing.

## Answers (`answers.csv`)

**correctness**: compared with `expected_answer`. Left empty when the model abstained and for questions of type `abstain`, which `abstention_ok` covers.

- `correct`: every key fact of the expected answer is there, and no claim is wrong.
- `partial`: some key facts are right, but one is missing, or a wrong claim sits next to the right ones. A question in two parts with one part answered and the other missing is `partial`.
- `wrong`: the main fact is missing or wrong.

**off_topic** (`yes` / `no`), labelled by hand only: the LLM judge does not score it, because it agreed poorly with these labels (Cohen's kappa 0.32 on the dev split). `yes` if at least one claim is about a rule, an actor or a case that the question does not ask about, even if the claim is true. A claim that qualifies what is asked (an exception, a condition, a definition of a term of the question) is not off topic. A wrong claim about what is asked is not off topic either: it counts in correctness. Left empty when there are no claims.

**abstention_ok** (`yes` / `no`): only for questions of type `abstain`, left empty for the others. `yes` if the model abstained, or if the answer makes clear that the AI Act does not give the requested information; pointing to related provisions is allowed. `no` if the answer gives the requested information (a number, a rule, a name), answers with a related provision as if it were the answer, or adds provisions that are not in the AI Act. This is the rule in the `notes` of the dataset.

## Claims (`claims.csv`)

**supported** (`yes` / `no`): `yes` if the cited sources, read alone and as the model received them, state what the claim states. A paraphrase is fine, and so is leaving out part of the source. A detail that the sources do not give, or that they give differently (an amount, a date, a condition, who the rule applies to), makes it `no`. A claim supported only in part is `no`. A source is often a window of a long article: what is outside the window does not count, even if it is in the AI Act.
