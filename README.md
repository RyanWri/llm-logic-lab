# LLM Logic Lab

## Overview

**LLM Logic Lab** is an experimental project aimed at investigating whether large language models (LLMs) possess logical reasoning capabilities. We developed a Python program that performs question answering on a set of sentences extracted from a text file, and we analyze the reasoning behind the answers using a chain-of-thought approach.

## Project Description

This project revolves around the following key tasks:

1. **Question Answering on Sentences:**  
   A Python program reads sentences from a text file and uses different LLMs (e.g., Mistral, Qwen2) to perform question answering.

2. **Chain-of-Thought Reasoning:**  
   For each sentence, the program generates detailed, step-by-step reasoning (chain of thought) that not only justifies the answer but also distinguishes between direct inferences and hidden assumptions.

3. **Comparative Analysis:**  
   The outputs from two different language models are compared to evaluate the depth and clarity of their reasoning. We discuss how each model handles reasoning tasks, identifies hidden assumptions, and where they might fall short.

4. **Extended Experiments:**  
   Additional experiments include:
   - Generating derived sentences by replacing words with "nonsense" terms.
   - Analyzing ambiguous sentences with tailored prompts.
   - Applying knowledge graphs to enhance reasoning.
   - Training the models with external datasets (e.g., ATOMIC, CONCEPTNet) and repeating experiments to compare performance improvements.

## Project Structure

The repository is organized into the following directories:

1. **src:**  
   Contains all code related to the project, including:
   - Model handlers
   - Fine-tuning scripts
   - Modules for managing entities and tasks

2. **data:**  
   Stores all sentence-related files used for analysis and testing.

3. **output:**  
   Contains model responses and generated reasoning chains.

4. **conclusions:**  
   Includes analysis reports, research findings, and discussions on the results.

5. **knowledge graph:**  
   Holds files and scripts related to RAG (retrieval-augmented generation) analysis and model node visualization.

## Implementation Details

- **Language:** Python  3.10
- **Libraries & Tools:**  
  - APIs for interacting with various LLMs (e.g., Mistral, Qwen2)
  - Standard libraries for file handling and threading
  - Data processing and analysis modules
  - Ollama custom logic for on permise models

## How to Run

1. **Install Dependencies:**  
   Ensure you have Python installed, then run:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python src/main.py 
   ```

2. The script will:
* Read sentences from the data directory.
* Query multiple LLMs for answers and generate chain-of-thought reasoning.
* Compare outputs and annotate reasoning steps.

## Conclusion
### please read **final-conclusion-nlp.docx**.
The Conclusions section (found in the conclusions directory) is the highlight of this project<br>It provides an extensive deep dive into our methodology, detailed experimental outcomes, and comprehensive reasoning comparisons between different LLMs. <br>We highly recommend reading this document for a thorough understanding of our analysis.



## Results and Analysis
The project produces several outputs:

### Detailed Reasoning Chains:
For every sentence, an in-depth chain-of-thought is generated that outlines the logical process behind the answer.

### Comparative Reports:
Side-by-side comparisons of reasoning from two LLMs, highlighting differences in handling assumptions and direct inferences.

### Extended Experiment Results:
Analysis of modified sentences, ambiguous prompt resolutions, knowledge graph enhancements, and the effects of fine-tuning with external datasets.

### Future Work
Future enhancements may include:
* Expanding the range of LLMs evaluated.
* Integrating additional datasets for training and analysis.
* Developing a web-based interface for interactive exploration of reasoning chains.
* Refining algorithms to better detect and classify hidden assumptions.
