template1 = """\
=== BEGIN PASSAGE ===
{passage}
=== END PASSAGE ===

{prefix} {question} Answer must be a short phrase."""

template2 = """\
Passage: {passage}
Question: {question}

{prefix} what is the correct answer?"""

template3 = """\
Read the following passage:
{passage}

{prefix} please answer this question: {question}"""

template4 = """\
Here is a passage:
{passage}

Based on the above, {prefix} {question}"""

template5 = """\
Given this passage:
{passage}

{prefix} could you answer: {question}"""

template6 = """\
Consider the following text:
{passage}

{prefix} respond to this question: {question}"""

template7 = """\
Text passage:
{passage}

Question: {question}
{prefix} what's your answer?"""

template8 = """\
From this passage:
{passage}

{prefix} please tell me: {question}"""

template9 = """\
After reading:
{passage}

{prefix} I'd like to know: {question}"""

template10 = """\
Examine this passage:
{passage}

{prefix} can you answer: {question}"""

templates = [template1, template2, template3, template4, template5, template6, template7, template8, template9, template10]