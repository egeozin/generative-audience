import pandas as pd
import numpy as np
from ast import literal_eval

import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import get_embedding
import os

openai.api_key = os.getenv('OPENAI_API_KEY')
print(openai.api_key)

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

encoding = tiktoken.get_encoding(embedding_encoding)
# omit reviews that are too long to embed

simpler_df = pd.read_csv("embeds_upworthy_experiment_simpler.csv")

def main():
    # Add a column with the number of tokens for each headline
    simpler_df["n_tokens"] = simpler_df.headline.apply(lambda x: len(encoding.encode(x)))
    filter_df = simpler_df[simpler_df.n_tokens <= max_tokens]
    len(filter_df)

    # Filter out headlines that are too long to embed
    filter_df = simpler_df[simpler_df.n_tokens <= max_tokens]

    # Check if the CSV file already exists
    if os.path.exists("ablation_data_openai_ada_002.csv"):
        # If it does, load it and find out where we left off
        filter_df = pd.read_csv("ablation_data_openai_ada_002.csv")
        start_idx = filter_df['embedding'].notna().sum()
        print(f"Resuming from index {start_idx}")
    else:
        # If it doesn't, initialize the 'embedding' column and start from the beginning
        filter_df["embedding"] = np.nan
        start_idx = 0

    # Get embeddings for each headline, starting from where we left off
    for i, row in filter_df.iloc[start_idx:].iterrows():
        try:
            embedding = get_embedding(row['headline'], engine=embedding_model)
            filter_df.at[i, "embedding"] =str(embedding)
        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue

        # Save the results to a CSV file every 100 steps
        if i % 100 == 0:
            filter_df.to_csv("ablation_data_openai_ada_002.csv", index=False)

    # Save the final results to a CSV file
    filter_df.to_csv("ablation_data_openai_ada_002.csv", index=False)

if __name__ == "__main__":
    main()