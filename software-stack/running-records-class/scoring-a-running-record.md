# Scoring a Running Record

## Repetitions

When we score a running record, we are often concerned with repetition. These repetitions occur when a child repeats word\(s\) that came earlier in the text. There are multiple types of repetitions that we need to record based on the child's reading.

#### Repetition of a Correct Response

For example, let's assume that the correct text is **It is a rainy day**. If the child says **It is a rainy, rainy day**, then we need to record a repetition in the form of the child repeating the word rainy multiple times. In this case, the child isn't saying the wrong text, but they are repeating correct text.

```text
Output: ["It", "is", "a", "rainy", "REPETITION", "day"]
```

#### Repetition of a Substitution

For example, let's assume that the correct text is **I need my coat**. If the child says **I need my jacket, jacket**, then we need to record a repetition in the form of the child repeating the substitution. 

```text
Output: ["I", "need", "my", "SUBSTITUTION", "REPETITION"]
```

#### Multiple Repetitions of a Correct Response

For example, let's assume that the correct text is **It is a rainy day**. If the child says **It is a rainy, rainy, rainy day** then we need to record multiple repetitions in the form of the child repeating the correct text multiple times.

```text
Output: ["It", "is", "a", "rainy", "REPETITION", "REPETITION", "day"]
```

#### Multiple Repetitions of a Substitution

For example, let's assume that the correct text is **I need my coat**. If the child says **I need my jacket, jacket, jacket**, then we need to record a repetition in the form of the child repeating the substitution multiple times. 

```text
Output: ["I", "need", "my", "SUBSTITUTION", "REPETITION", "REPETITION"]
```

#### Repetition of More Than One Word

For example, let's assume that the correct text is **It is a rainy day**. If the child says **It is a it is a rainy day**, then we need to record a repetition in the form of a child repeating multiple words, because they are repeating the phrase "It is a."

```text
Output: ["It", "is", "a", "REPETITION", "REPETITION", "REPETITION", "rainy", "day"]
```

#### Repetition of More Than One Word With a Self-Correction on the Rereading

This example will be a little more complicated, but let's assume the correct text is "I need my coat. Is it in the closet?" If the child says "I need my coat. Is it in my is it in the closet?" then we need to note that they used the self-correction to repeat.

```text
Output: ["I", "need", "my", "coat", "Is", "it", "in", "INSERTION", "REPETITION", "REPETITION", "REPETITION", "the", "closet"]
```

