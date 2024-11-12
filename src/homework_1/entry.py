import nltk
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.lang.he import Hebrew
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
nltk.download('punkt')
nltk.download('punkt_tab')

def english():
    # Initialize the English language model
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    
    text = "My name is Israel Israeli, and I live in Israel, in Tel-Aviv."
    # Use the nlp pipeline to process text before tokenization
    doc = nlp(text)
    tokens = [token.text for token in doc]  # Extract text from tokens
    print("English Tokens:", tokens)

def hebrew():
    # Initialize Hebrew tokenizer
    text_hebrew = "שלום כיתה א"
    nlp = Hebrew()
    tokenizer = Tokenizer(nlp.vocab)
    
    doc = nlp(text_hebrew)
    tokens = [token.text for token in doc]
    print("Hebrew Tokens:", tokens)

    # Using Treebank Tokenizer and Detokenizer on English example text
    text = "My name is Israel Israeli, and I live in Israel, in Tel-Aviv."
    t = TreebankWordTokenizer()
    d = TreebankWordDetokenizer()
    
    toks = t.tokenize(text)
    print("Treebank Tokenizer Output:", toks)
    
    untoks = d.detokenize(toks)
    print("Treebank Detokenizer Output:", untoks)

if __name__ == '__main__':
    # Main text to test word_tokenize from nltk
    text = """My name is Israel Israeli, and I live in Israel, in Tel-Aviv. I'm studying Computer
    Science in Afeka college, and although I can't say that I enjoy every moment of studying, 
    I feel like I'm learning much and acquiring programming skills that will help me find 
    a good job as a waiter... joking"""

    # NLTK tokenization example
    res = nltk.word_tokenize(text)
    print("NLTK Tokenization:", res)

    # Run tokenization functions
    english()
    hebrew()
