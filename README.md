# AI Assistant Performance Analyzer

Did ChatGPT really get worse over time after each release, like lots of people suspected?

This tool takes your downloaded ChatGPT conversations and asks ChatGPT to rate the first
interaction in each conversation according to a few of different metrics.

It then generates a graph that shows how it has changed over time.

This was a quick and dirty project to try out zed and cursor. They wrote most of the code, so
the code quality is... well, good enough.

## Details

For rating the conversation, it currently uses gpt-4o-mini. We only look at the first interaction because that's the most bang for the buck in terms of cost.
Ratings are normalized relative to the difficulty of the prompt, which is also assessed by gpt-4o-mini.
Cost was about $1.20 for ~3200 conversations.

The ratings are cached (in output/ratings_cache.csv) so if something goes wrong, you can just run the tool
again and it will pick up and continue where it was stopped.

## Requirements

- Python 3.7+
- OpenAI API key (for generating quality ratings)
- Required Python packages:
  - pandas
  - plotly
  - pydantic
  - tiktoken
  - openai

## Setup

1. Clone this repository
2. Install required packages:
   ```
   pip install pandas plotly pydantic tiktoken openai
   ```
3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

1. Download your data from ChatGPT (top right -> Settings -> Data Controls -> Export your data)
2. Put that file into the working directory for the code (same as where openai_analyzer.py is)
3. Run the analysis script:
   ```
   python openai_analyzer.py
   ```
4. View `output/index.html`

## Additional Output

Other than index.html and the files it links to (graph.html, ratings/*), the tool generates:

- `organized_conversations.json`: Processed and organized conversation data
- `ratings_cache.csv`: Cached quality ratings

## Notes

- The project uses the OpenAI API to generate quality ratings, which may incur costs
- A dry run mode is available to test the workflow without making API calls
- The analysis can be interrupted and resumed, as it caches processed data

## License

Do whatever you want with the code. But please credit me (@outscape on X) for at least inspiring it when you post about it.
