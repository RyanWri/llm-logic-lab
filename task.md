הפרויקט ניתן לביצוע ביחידים / זוגות / שלשות.
יש לבחור את אחד הנושאים המתוארים להלן
ולבצע עבורו את כל האנליזות הנדרשות.
תוצרי הפרויקט:
*
GITHUB
הכולל את כל קבצי הפייתון + תיעוד שלהם
WORD * קובץ
המתאר את המתודולוגיה + התוצאות
________________________________________________________
1 פרויקט
:
האם למודל שפה יש לוגיקה?

1.) Write a python program that performed Question Answering on sentences
(embedded in a text file) by using different LLMs

2.) The sentences in the text file are:

a. John couldn't find his glasses while they were on his head.

b. After the rain, Sarah grabbed her umbrella before leaving the office.

c. The coffee was too hot to drink, so I added an ice cube.

d. Tom can't go to his sister's wedding because he's studying abroad in
Japan.

e. She put the groceries away before the ice cream melted.

f. The plant died because Jenny forgot to open the curtains for a week.

3.) To each of these sentences, write as much reasoning as possible. For example, a
common-sense chain for the 4th sentence might be:

• Weddings are events that happen at specific times and places

• Japan is far from (assumed location)

• Travel from Japan is expensive and time-consuming

• Students typically have academic commitments

• Missing classes/exams can have consequences

• Therefore, Tom's location and commitments prevent attendance

4.) Compare the results achieved from 2 different language models (e.g. gpt3, mt5,
bert). Describe the results and discuss the differences.

5.) For each sentence in the chain – write whether there is a hidden assumption or
it is a direct reasoning from the given sentence.

6.) Find a sentence that at least one of the models fails to find the reasoning chain.
Explain why it fails.

7.) Choose one of the sentences from section (2) and generate from it a set of
derived sentences, that one or more words are replaced by "nonsense" words.
Apply task 4 to these sentences. Write the results and explain them. Example of
such derived sentences from the given © sentence:

a. The mnbm was too hot to drink, so I added an ice cube.

b. The coffee was too ksdf to drink, so I added an ice cube.

c. The coffee was too hot to ksafd, so I added an ice cube.

d. The coffee was too hot to drink, so I kdjfs an ice cube.

e. The coffee was too hot to drink, so I added a sdflsd.

f. The mnbm was too hot to ksafd, so I added an ice cube.

g. The coffee was too hot to ksafd, so I kdjfs an ice cube.

h. The mnbm was too ksdf to ksafd, so I kdjfs an sdflsd.

8.) The following sentence might have two different meanings: "the chicken is
ready to eat".

a.) Propose a prompt or a set of prompts that will let us understand which
meaning a language model selects.

b.) Apply this prompt to 2 different language models and compare and explain
their results.

c.) Find another ambiguity that the two models relate the same sentence
differently.

d.) Can you propose an algorithmic solution to be applied in a case of an
ambiguity?

9.) Choose 3 sentences from (2) and apply to them knowledge graph. Explain how
can it used to improve the reasoning?

10.) Train the models you choose in (4) with ATOMIC / CONCEPTNet dataset, and
repeat steps 4, 6 and 7. Explain the results.