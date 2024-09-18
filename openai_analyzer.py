import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime
import html
import openai
import time
import csv
import os
import random
import tiktoken
import sys
from openai import OpenAIError
from pydantic import BaseModel, Field
from openai import OpenAI

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import numpy as np



# Dry run mode
DRY_RUN = False  # Set to False for actual API calls
MAX_RATED_COUNT = 2000000

# Replace BASE_DIR with INPUT_DIR and OUTPUT_DIR
INPUT_DIR = "input"  # Directory for input files
OUTPUT_DIR = "output"  # Directory for output files
CACHE_FILE = os.path.join(OUTPUT_DIR, 'ratings_cache.csv')


# Define dimensions only once
ASSISTANT_DIMENSIONS = [
    ("overall", "Overall quality of the response"),
    ("relevance", "How well the response addresses the user's query or task"),
    ("accuracy", "Correctness of information provided"),
    ("coherence", "Logical flow and consistency of the response"),
    ("completeness", "How thoroughly the response addresses all aspects of the query"),
    ("clarity", "How easy the response is to understand"),
    ("conciseness", "Efficiency in conveying information without unnecessary verbosity"),
    ("helpfulness", "How useful the response is in solving the user's problem or answering their question"),
    ("safety", "Avoidance of harmful, unethical, or biased content"),
    ("creativity", "Novel or innovative aspects of the response"),
    ("language_quality", "Proper grammar, spelling, and language use"),
    ("task_completion", "How well the response accomplishes the requested task"),
    ("contextual_understanding", "How well the response considers the context of the conversation"),
    ("non_sycophancy", "Ability to provide honest, critical feedback when appropriate instead of always agreeing")
]

USER_DIMENSIONS = [
    ("complexity", "The level of difficulty or intricacy of the user's query"),
    ("domain_specificity", "How specialized or niche the topic of the query is"),
    ("ambiguity", "How clear or unclear the user's request is"),
    ("abstraction_level", "Whether the query deals with concrete or abstract concepts"),
    ("contextual_requirements", "How much context is needed to fully understand and address the query"),
    ("linguistic_challenge", "The level of language proficiency required to understand and respond to the query"),
    ("cognitive_demand", "The level of cognitive processing required to address the query")
]

ALL_DIMENSIONS = [("a_" + dim[0], dim[1]) for dim in ASSISTANT_DIMENSIONS] + [("u_" + dim[0], dim[1]) for dim in USER_DIMENSIONS]


class QualityRating(BaseModel):
    pass

for prefix, dimensions in [("a_", ASSISTANT_DIMENSIONS), ("u_", USER_DIMENSIONS)]:
    for dim, _ in dimensions:
        setattr(QualityRating, f"{prefix}{dim}", Field(..., ge=0, le=100))


@dataclass
class Message:
    id: str
    author_role: str
    content: str

@dataclass
class Conversation:
    id: str
    title: str
    create_time: float
    update_time: float
    messages: List[Message]


def process_conversations(file_path: str) -> Dict[str, Dict[Tuple[str, str], List[Conversation]]]:
    with open(file_path, 'r') as f:
        data = json.load(f)

    organized_data = {}

    conversations = data if isinstance(data, list) else [data]

    for conversation in conversations:
        model = conversation.get('default_model_slug', 'Unknown')
        if model is None:
            model = 'Unknown'

        if ("gpt" not in model.lower()) and model.lower() != "unknown":
            print(f"SKIPPING CONVERSATION WITH MODEL: {model.upper()} - NOT A GPT MODEL")
            continue

        messages = []
        first_user_message = None
        first_assistant_response = None

        about_model = ''
        about_user = ''

        for msg in conversation.get('mapping', {}).values():

            message = msg.get('message')

            if message and message.get('metadata') and message['metadata'].get('user_context_message_data'):
                about_user = message['metadata']['user_context_message_data'].get('about_user_message', '')
                about_model = message['metadata']['user_context_message_data'].get('about_model_message', '')

            if message and message.get('content'):
                content = message['content']
                if isinstance(content, dict):
                    # Skip image messages
                    if 'content_type' in content and content['content_type'] == 'image_asset_pointer':
                        if not first_user_message:
                            # If this is the first user message and it's an image, skip this conversation
                            break
                        continue
                    content_text = ''.join(str(part) for part in content.get('parts', []))
                else:
                    content_text = str(content)

                if content_text.strip():  # Skip blank messages
                    msg_obj = Message(
                        id=message['id'],
                        author_role=message['author']['role'],
                        content=content_text
                    )
                    messages.append(msg_obj)

                    if not first_user_message and msg_obj.author_role == 'user':
                        first_user_message = msg_obj
                    elif first_user_message and not first_assistant_response and msg_obj.author_role == 'assistant':
                        first_assistant_response = msg_obj
                        break  # We have what we need, stop processing messages

        if first_user_message and first_assistant_response:
            conv = Conversation(
                id=conversation['id'],
                title=conversation.get('title', ''),
                create_time=conversation.get('create_time', 0),
                update_time=conversation.get('update_time', 0),
                messages=[first_user_message, first_assistant_response]
            )

            key = (about_user, about_model)

            if model not in organized_data:
                organized_data[model] = {}
            if key not in organized_data[model]:
                organized_data[model][key] = []
            organized_data[model][key].append(conv)

    return organized_data

def load_ratings_cache(cache_file: str) -> Dict[Tuple[str, str], Dict[str, int]]:
    ratings_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamp = row['timestamp']
                table_id = row['table_id']
                ratings = {k: int(float(v)) for k, v in row.items() if k not in ['timestamp', 'table_id', 'model']}
                ratings_cache[(timestamp, table_id)] = ratings
    return ratings_cache

def save_rating_to_cache(cache_file: str, timestamp: str, table_id: str, model: str, ratings: Dict[str, int]):
    file_exists = os.path.exists(cache_file)
    with open(cache_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'table_id', 'model', 'user_tokens', 'assistant_tokens'] + [dim[0] for dim in ALL_DIMENSIONS])
        writer.writerow([timestamp, table_id, model, ratings['user_tokens'], ratings['assistant_tokens']] + [ratings[dim[0]] for dim in ALL_DIMENSIONS])

def num_tokens_from_string(string: str, model_name: str) -> int:
    if model_name == 'Unknown':
        # Choose a default tokenizer or use a simple method
        return len(string.split())  # Simple word count as a fallback

    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

def calculate_cost(tokens):
    return (tokens / 1_000_000) * 0.25  # $0.25 per 1M tokens

cumulative_cost = 0

def get_quality_rating(user_message: str, assistant_response: str, model_name: str, timestamp: str, table_id: str) -> Dict[str, int]:
    global cumulative_cost
    user_tokens = num_tokens_from_string(user_message, model_name)
    assistant_tokens = num_tokens_from_string(assistant_response, model_name)

    ratings = {
        'user_tokens': user_tokens,
        'assistant_tokens': assistant_tokens
    }

    try:
        if DRY_RUN:
            ratings.update({dim[0]: random.randint(0, 100) for dim in ALL_DIMENSIONS})
        else:
            client = OpenAI()
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are an AI quality rater.
Your task is to rate the difficulty of the user's prompt and the quality of an AI assistant's response.
Provide integer ratings from 0 to 100 for multiple dimensions,
where 0 is easy prompt/very poor response and 100 is difficult prompt / excellent response."""},

                    {"role": "user", "content": f"""User message: {user_message}

Assistant response: {assistant_response}

Please rate the following dimensions with an integer between 0 and 100.

For the assistant's response (prefix with 'a_'):
{', '.join([f"{dim[0]} ({dim[1]})" for dim in ASSISTANT_DIMENSIONS])}

For the user's prompt difficulty (prefix with 'u_'):
{', '.join([f"{dim[0]} ({dim[1]})" for dim in USER_DIMENSIONS])}"""}
                ],
                response_format=QualityRating,
            )

            parsed_ratings = completion.choices[0].message.parsed
            ratings.update({k: int(v) for k, v in parsed_ratings.dict().items()})

            # Calculate and print cost
            api_tokens = completion.usage.total_tokens
            call_cost = calculate_cost(api_tokens)
            cumulative_cost += call_cost
            print(f"API call cost: ${call_cost:.6f}")
            print(f"Cumulative cost: ${cumulative_cost:.6f}")

        # Save the full messages to an HTML file
        save_messages_to_html(user_message, assistant_response, timestamp, table_id, ratings)

    except OpenAIError as e:
        print(f"OpenAI API error: {str(e)}")
        raise

    return ratings

def generate_rating_filename(timestamp: str, table_id: str) -> str:
    return f"ratings/{timestamp}_{table_id}.html"

def save_messages_to_html(user_message: str, assistant_response: str, timestamp: str, table_id: str, ratings: Dict[str, int]):
    os.makedirs(os.path.join(OUTPUT_DIR, 'ratings'), exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, generate_rating_filename(timestamp, table_id))

    # Function to determine color based on rating
    def get_color(rating):
        try:
            rating_value = int(rating)
            return f'rgb({255-int(rating_value*2.55)}, {int(rating_value*2.55)}, 0)'
        except (ValueError, TypeError):
            # If conversion fails, return a default color
            return 'rgb(128, 128, 128)'  # Gray color as fallback

    # Create table rows for ratings
    rating_rows = ''
    for dim, desc in ALL_DIMENSIONS:
        key = dim
        if key in ratings:
            value = ratings[key]
        else:
            value = "N/A"  # or some default value
        color = get_color(value)
        rating_rows += f'<tr><td>{dim}</td><td>{desc}</td><td style="background-color: {color}">{value}</td></tr>'

    html_content = f"""
    <html>
    <head>
        <title>Conversation - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
            .message {{ margin-bottom: 20px; padding: 10px; border-radius: 5px; }}
            .user {{ background-color: #e6f3ff; }}
            .assistant {{ background-color: #f0f0f0; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Conversation - {timestamp}</h1>
        <div class="message user">
            <h2>User Message:</h2>
            <pre>{html.escape(user_message)}</pre>
        </div>
        <div class="message assistant">
            <h2>Assistant Response:</h2>
            <pre>{html.escape(assistant_response)}</pre>
        </div>
        <div class="ratings">
            <h2>Ratings:</h2>
            <table>
                <tr><th>Dimension</th><th>Description</th><th>Rating</th></tr>
                {rating_rows}
            </table>
        </div>
    </body>
    </html>
    """

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

def shorten_string(s, length):
    return f'{s[:length]}... [{len(s)}]'


def generate_index(organized_conversations, ratings_cache):
    html_report = """<html><head>
<style>body{font-family:Arial,sans-serif;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ddd;padding:8px;text-align:left;} th{background-color:#f2f2f2;}</style>
</head>
<body>
<h1>ChatGPT Conversations</h1>
<p>
Does it seem like ChatGPT got worse over time? Now you can see for yourself.
<p>
This tool only looks at the first interaction in each conversation.
If the prompt included an image, the conversation is skipped.
Each conversation is rated - click on the link in the first column.
</p>
<p>
Conversations are organized by model and your user/system prompt.
</p>
<p>
See <a href="graphs.html">these plots with your results.</a>
</p>
<h1>Conversation Analysis Report</h1>
<h2>Index</h2>
<ul>"""
    for model in organized_conversations:
        about_id = 0
        for _ in organized_conversations[model]:
            about_id += 1
            table_id = f"{model}_{about_id}"
            html_report += f'<li><a href="#{table_id}">{table_id}</a></li>'
    html_report += '</ul>'

    for model, conversations in organized_conversations.items():
        about_id = 0
        for (about_user, about_model), convs in conversations.items():
            about_id += 1
            table_id = f"{model}_{about_id}"
            html_report += f'<h2 id="{table_id}">Model: {table_id}</h2>'

            html_report += f'<h3>About User:</h3>'
            html_report += f'<p>{html.escape(about_user)}</p>'
            html_report += f'<h3>About Model:</h3>'
            html_report += f'<p>{html.escape(about_model)}</p>'

            html_report += '<table>'
            html_report += '<tr><th>Timestamp</th><th>Prompt</th><th>Response</th><th>Ratings</th></tr>'

            for conv in convs:
                if len(conv.messages) >= 2:
                    user_message = conv.messages[0].content
                    assistant_response = conv.messages[1].content

                    timestamp = datetime.fromtimestamp(conv.create_time).strftime('%Y%m%d_%H%M%S')
                    rating_file_url = generate_rating_filename(timestamp, table_id)

                    html_report += f'<tr>'
                    html_report += f'<td><a href="{rating_file_url}" target="_blank">{timestamp}</a></td>'

                    html_report += f'<td>{shorten_string(user_message, 40)}</td>'
                    html_report += f'<td>{shorten_string(assistant_response, 40)}</td>'

                    if (timestamp, table_id) in ratings_cache:
                        ratings = ratings_cache[(timestamp, table_id)]
                        html_report += '<td>'
                        for dim in ['a_overall', 'u_complexity']:
                            value = ratings.get(dim, 'N/A')
                            color = get_color(value)
                            html_report += f'<span style="background-color: {color}">{dim}: {value}</span><br>'
                        html_report += '</td>'
                    else:
                        html_report += '<td>Not rated yet</td>'

                    html_report += '</tr>'

            html_report += '</table>'

    html_report += '</body></html>'

    path = os.path.join(OUTPUT_DIR, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    print(f"Report generated: {path}")

    return

def process_uncached_ratings(organized_conversations, ratings_cache, cache_file):
    global cumulative_cost
    skip_counter = 0

    for model, conversations in organized_conversations.items():
        print(f"Processing model: {model}")
        about_id = 0
        for (about_user, about_model), convs in conversations.items():
            processed_count = 0

            about_id += 1
            print(f"  Processing conversation group: {about_id}")
            for conv in convs:
                if len(conv.messages) < 2:
                    print(f"    Skipping conversation due to insufficient messages")
                    continue

                user_message = conv.messages[0].content
                assistant_response = conv.messages[1].content
                assert conv.messages[0].author_role == 'user', "First message should be from user"
                assert conv.messages[1].author_role == 'assistant', "Second message should be from assistant"

                if not isinstance(user_message, str):
                    print(f"    Skipping conversation: User message is not a string (type: {type(user_message)})")
                    continue

                if 'image_asset_pointer' in user_message:
                    #print(f"    Skipping conversation: User message starts with image content")
                    skip_counter += 1
                    continue


                timestamp = datetime.fromtimestamp(conv.create_time).strftime('%Y%m%d_%H%M%S')
                table_id = f"{model}_{about_id}"

                if (timestamp, table_id) not in ratings_cache:
                    print(f"    Processing message: {table_id}")
                    ratings = get_quality_rating(user_message, assistant_response, model, timestamp, table_id)
                    save_rating_to_cache(cache_file, timestamp, table_id, model, ratings)
                    print(f"    Ratings calculated and saved for {table_id}")
                    processed_count += 1
                    #else:
                    #print(f"    Using cached ratings for {table_id}")

                if processed_count >= MAX_RATED_COUNT:
                    print(f"\nReached maximum number of rated conversations ({MAX_RATED_COUNT})")
                    print(f"Total cost: ${cumulative_cost:.6f}")
                    continue

    print(f"\nProcessing complete. Total conversations processed: {processed_count}")
    print(f"Total conversations skipped due to image content: {skip_counter}")
    print(f"Total cost: ${cumulative_cost:.6f}")


def get_color(rating):
    try:
        rating_value = int(rating)
        return f'rgb({255-int(rating_value*2.55)}, {int(rating_value*2.55)}, 0)'
    except (ValueError, TypeError):
        return 'rgb(128, 128, 128)'  # Gray color as fallback

def save_organized_conversations(organized_conversations, output_file):
    def tuple_key_to_str(obj):
        if isinstance(obj, dict):
            return {str(key) if isinstance(key, tuple) else key: tuple_key_to_str(value)
                    for key, value in obj.items()}
        elif isinstance(obj, list):
            return [tuple_key_to_str(element) for element in obj]
        elif isinstance(obj, (Conversation, Message)):
            return obj.__dict__
        else:
            return obj

    serializable_data = tuple_key_to_str(organized_conversations)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, default=str, ensure_ascii=False)


def create_multi_line_plot(df, x, y_list, title):
    fig = go.Figure()
    df_sorted = df.sort_values(by=x)
    for y in y_list:
        if y in df.columns:  # Only plot if the column exists
            # Calculate the rolling average
            y_rolling = df_sorted[y].rolling(window=500, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=df_sorted[x], y=y_rolling, mode='lines', name=y))
    fig.update_layout(title=title, height=600, xaxis_title="Time", yaxis_title="Value")
    return fig

def create_section_plots(df, table_id):
    plots = []

    # Non-normalized multi-line plot for all a_ values
    a_columns = [col for col in df.columns if col.startswith('a_')]
    plots.append(create_multi_line_plot(df, 'timestamp', a_columns, f'Assistant Metrics Over Time - {table_id}'))

    # Non-normalized multi-line plot for all a_ values
    u_columns = [col for col in df.columns if col.startswith('u_')]
    plots.append(create_multi_line_plot(df, 'timestamp', u_columns, f'User Prompt Metrics Over Time - {table_id}'))

    # Normalized multi-line plot for all a_ values divided by average prompt complexity (excluding creativity and safety)
    normalized_a_values = df[['timestamp'] + a_columns].copy()
    for col in a_columns:
        normalized_a_values[col] = normalized_a_values[col] / df['avg_prompt_complexity']
    plots.append(create_multi_line_plot(normalized_a_values, 'timestamp', a_columns, f'Normalized Assistant Metrics Over Time - {table_id}'))

    # Separate line plots for creativity, safety, and avg_assistant_quality (non-normalized and normalized)
    for metric in ['avg_assistant_quality', 'a_creativity', 'a_safety']:
        # Non-normalized plot
        plots.append(create_multi_line_plot(df, 'timestamp', [metric], f'{metric.capitalize()} Over Time - {table_id}'))

        # Normalized plot
        normalized_metric = df[metric] / df['avg_prompt_complexity']
        plots.append(create_multi_line_plot(pd.DataFrame({'timestamp': df['timestamp'], metric: normalized_metric}), 'timestamp', [metric], f'Normalized {metric.capitalize()} Over Time - {table_id}'))



    # Heatmap: Correlation between metrics
    corr_metrics = [col for col in df.columns if col.startswith('a_') or col.startswith('u_')]
    corr_matrix = df[corr_metrics].corr()
    fig = px.imshow(corr_matrix, title=f'Correlation Heatmap - {table_id}')
    fig.update_layout(height=800, width=800)
    plots.append(fig)

    return plots

def generate_plots():

    df = pd.read_csv(CACHE_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
    df['avg_assistant_quality'] = df[['a_overall', 'a_relevance', 'a_accuracy', 'a_coherence', 'a_completeness', 'a_clarity', 'a_conciseness', 'a_helpfulness', 'a_task_completion', 'a_contextual_understanding', 'a_non_sycophancy']].mean(axis=1)
    df['avg_prompt_complexity'] = df[['u_complexity', 'u_domain_specificity', 'u_ambiguity', 'u_abstraction_level', 'u_contextual_requirements', 'u_linguistic_challenge', 'u_cognitive_demand']].mean(axis=1)
    df['quality_to_complexity_ratio'] = df['avg_assistant_quality'] / df['avg_prompt_complexity']

    html = """
    <html>
    <head>
        <title>AI Assistant Performance Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; }
            .index { margin-bottom: 20px; }
            .index a { margin-right: 10px; }
        </style>
    </head>
    <body>
        <h1>AI Assistant Performance Analysis</h1>
        <div class="index">
            <h2>Index</h2>
    """

    # Generate index
    html += f'<a href="#all_data">All Data ({len(df)} data points)</a>'
    for table_id in df['table_id'].unique():
        df_table = df[df['table_id'] == table_id]
        html += f'<a href="#{table_id}">{table_id} ({len(df_table)} data points)</a>'

    html += "</div>"

    # All data section
    html += f'<h2 id="all_data">All Data ({len(df)} data points)</h2>'
    for plot in create_section_plots(df, "All Data"):
        html += plot.to_html(full_html=False, include_plotlyjs=False)

    # Sections for each table_id
    for table_id in df['table_id'].unique():
        df_table = df[df['table_id'] == table_id]
        html += f'<h2 id="{table_id}">{table_id} ({len(df_table)} data points)</h2>'
        for plot in create_section_plots(df_table, table_id):
            html += plot.to_html(full_html=False, include_plotlyjs=False)

    html += "</body></html>"

    with open("output/graphs.html", "w") as f:
        f.write(html)


organized_conversations = process_conversations(os.path.join(INPUT_DIR, 'openai/conversations.json'))

ratings_cache = load_ratings_cache(CACHE_FILE)

generate_index(organized_conversations, ratings_cache)

# Save organized data as pretty-printed JSON
organized_data_output = os.path.join(OUTPUT_DIR, 'organized_conversations.json')
save_organized_conversations(organized_conversations, organized_data_output)
print(f"Organized data saved to: {organized_data_output}")

process_uncached_ratings(organized_conversations, ratings_cache, CACHE_FILE)

# again, to update it with the new ratings
generate_index(organized_conversations, ratings_cache)

generate_plots()

print("All the ratings are in output/ratings_cache.csv if you want to do your own analysis.")
print(f"Look at {CACHE_FILE} for results.")
