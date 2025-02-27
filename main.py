import requests
from bs4 import BeautifulSoup
import json
import argparse
from googlesearch import search
import time
from llama_cpp import Llama

# Path to your local LLaMA model
model_path = "/Users/samaynandwana/Desktop/llama-3-8b-instruct.Q4_K_M.gguf"

# Initialize the LLaMA model
llm = Llama(model_path=model_path, n_ctx=6000)

# Argument Parser
parser = argparse.ArgumentParser(description="Scrape drug-related crime information.")
parser.add_argument("drug_name", type=str, help="The drug of interest.")
args = parser.parse_args()

drug_name = args.drug_name

def get_wikipedia_summary(drug_name):
    """Fetches a concise, factual Wikipedia summary for the given drug."""
    search_url = f"https://en.wikipedia.org/wiki/{drug_name.replace(' ', '_')}"
    try:
        response = requests.get(search_url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')

            summary_text = " ".join([p.get_text().strip() for p in paragraphs[:5]])  # Extract first 5 paragraphs
            prompt = f"Summarize this Wikipedia information in a concise, factual manner:\n\n{summary_text}"
            
            response = llm(prompt)
            return response["choices"][0]["text"].strip() if "choices" in response else "Error summarizing Wikipedia."
        return "Failed to retrieve Wikipedia summary."
    except Exception as e:
        return f"Error fetching Wikipedia summary: {str(e)}"

def get_news_links(drug_name):
    """Fetches recent news article links related to the drug and crime."""
    query = f"{drug_name} drug crime"
    news_links = []
    try:
        for url in search(query, num_results=10):
            news_links.append(url)
        return news_links
    except Exception as e:
        return [f"Error fetching news links: {str(e)}"]

def extract_full_text(news_links):
    """Extracts full content from all news articles."""
    articles = []
    for link in news_links:
        try:
            response = requests.get(link, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')

                full_text = " ".join([p.get_text().strip() for p in paragraphs])[:4000]
                articles.append({"url": link, "content": full_text})
        except Exception as e:
            articles.append({"url": link, "content": f"Error fetching content: {str(e)}"})
    return articles

def summarize_articles_with_llama(articles):
    """Summarizes multiple news articles using LLaMA."""
    summaries = []
    
    for article in articles:
        if len(article["content"]) < 100:  # Ignore short, irrelevant content
            continue
        
        prompt = f"Summarize this news article concisely, keeping only factual and relevant points:\n\n{article['content']}"
        
        response = llm(prompt)
        summary_text = response["choices"][0]["text"].strip() if "choices" in response else "Error summarizing article."
        
        # Ensure the summary makes sense
        if len(summary_text) > 30 and "error" not in summary_text.lower():  
            summaries.append({"url": article["url"], "summary": summary_text})
    
    return summaries

# Execute Scraping
print(f"Scraping data for: {drug_name}")
wikipedia_summary = get_wikipedia_summary(drug_name)
news_links = get_news_links(drug_name)
articles = extract_full_text(news_links)
news_summaries = summarize_articles_with_llama(articles)

# Save results
drug_info = {
    "Drug": drug_name,
    "Wikipedia Summary": wikipedia_summary,
    "News Articles": news_summaries
}

with open("drug_info_scraped.json", "w") as json_file:
    json.dump(drug_info, json_file, indent=4)

print("Scraping completed. Data saved to drug_info_scraped.json")
