import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from llama_cpp import Llama
import json
import textwrap

# Load LLaMA 3 model (ensure correct path)
model_path = "/Users/samaynandwana/Desktop/llama-3-8b-instruct.Q4_K_M.gguf"
llm = Llama(model_path=model_path)

# Provided articles
articles = [
    {
        "id": 1,
        "title": "Fort Wayne Man Sentenced to 84 Months",
        "text": """FORT WAYNE – Yesterday, Hamed A. Martin, 42 years old, of Fort Wayne, Indiana was sentenced by United States District Court Chief Judge Holly Brady after pleading guilty to distributing methamphetamine, announced Acting United States Attorney Tina L. Nommay.
        
        Martin was sentenced to 84 months in prison followed by 4 years of supervised release.
        
        According to documents in the case, in July 2022, Martin distributed methamphetamine on several occasions. A search warrant executed at his residence in August 2022, resulted in the recovery of a firearm along with evidence of drug distribution.
        
        This case was investigated by the Federal Bureau of Investigation’s Fort Wayne Safe Streets Gang Task Force, which includes the FBI, the Indiana State Police, the Allen County Sheriff’s Department, and the Fort Wayne Police Department. Also assisting in the investigation were the Drug Enforcement Administration and the DEA’s North Central Laboratory. The case was prosecuted by Assistant United States Attorney Stacey R. Speith."""
    },
    {
        "id": 2,
        "title": "Former Prison Guard Sentenced for Smuggling Drugs",
        "text": """BOSTON – A Virginia man was sentenced yesterday for conspiring to distribute controlled substances and launder drug proceeds with co-conspirators in Massachusetts and Virginia.

        Kenneth J. Owen, 24, of Charlotte Court House, Va., was sentenced by U.S. District Court Chief Judge F. Dennis Saylor IV to 21 months in prison, to be followed by three years of supervised release. In September 2024, Owen pleaded guilty to one count of conspiracy to distribute and possess with intent to distribute MDMA and buprenorphine and two counts of money laundering conspiracy.
        
        In December 2019 and January 2020, Owen conspired with Sathtra Em, a Lowell resident, and Michael Mao, an inmate at the Buckingham Correctional Center in Dillwyn, Va., to smuggle MDMA and buprenorphine in the form of Suboxone and generic Suboxone sublingual films into the prison. At the time, Owen was working as a correctional officer at Buckingham.
        
        As part of the conspiracy, Em mailed the drugs to Owen’s residence and paid him $1,600 in bribes to deliver the drugs and other contraband to Mao in the prison. Mao then sold the smuggled drugs to other inmates at Buckingham and Em collected the drug debts on behalf of Mao in the same Cash App accounts she used to pay the bribes to Owen. Owen used a Cash App account with the name “Carlos” to receive the bribes from Em, and he cashed out the funds to his bank account within minutes of receiving them.
        
        Em and Mao previously pleaded guilty to their roles in the conspiracy. In August 2024, Em was sentenced to 21 months in prison to be followed by three years of supervised release. Mao was sentenced to 121 months in prison to be followed by four years of supervised release, in November 2024."""
    },
    {
        "id": 3,
        "title": "Drug Trafficker with Mexican Cartel Sentenced",
        "text": """LAREDO, Texas – A 37-year-old man has been sentenced for conspiring to distribute a large quantity of marijuana, announced U.S. Attorney Nicholas J. Ganjei.

        Gavino Cadena pleaded guilty Nov. 10, 2022.

        U.S. District Judge Diana Saldana has now ordered Cadena to serve a total of 194 months in federal prison to be followed by five years of supervised release. In handing down the sentence, the court considered Cadena’s extensive criminal record, including his involvement with Cartel del Noreste (CDN) and the Tango Blast gang. Records also showed that while in custody awaiting sentencing in this case, Cadena was involved in numerous altercations with rival gang members such as Hermano Pistoleros Latinos, including incidents involving weapons.

        The court found Cadena to be a leader/organizer within the drug trafficking organization. He coordinated the drug loads, paid co-conspirators for their involvement and reported directly to cartel leaders in Mexico. Cadena was held responsible for organizing the offloading and transport of more than 8,000 pounds of marijuana from multiple tractor trailers in Laredo that had been imported from Mexico.

        Throughout the course of this multi-year investigation, which includes two related indictments, authorities seized more than 17 tons of marijuana valued at approximately $16.4 million."""
    }
]

# Function to generate embeddings using LLaMA 3
def get_embedding(text):
    """Generates embeddings for text, chunking if necessary."""
    prompt = f"Generate a 512-dimensional numerical embedding for the following text in JSON format:\nText: {text[:450]}\nOutput: {{\"embedding\": [list of 512 numbers]}}"

    response = llm(prompt, stop=["Output:"], max_tokens=800)

    try:
        json_str = response["choices"][0]["text"].strip()
        json_str = json_str[json_str.index("{"):]  # Extract JSON portion
        embedding_dict = json.loads(json_str)

        if "embedding" in embedding_dict and isinstance(embedding_dict["embedding"], list):
            return np.array(embedding_dict["embedding"])
        else:
            raise ValueError("Invalid JSON structure for embeddings.")

    except Exception as e:
        print(f"Error processing embedding: {e}")
        return np.zeros(512)  # Fallback to zero vector

# Create DataFrame
df = pd.DataFrame(articles)

# Compute embeddings for each article
df["embedding"] = df["text"].apply(get_embedding)

# Convert embeddings to FAISS index
embedding_dim = len(df["embedding"].iloc[0])
index = faiss.IndexFlatL2(embedding_dim)

# Normalize and add vectors
normalized_embeddings = normalize(np.vstack(df["embedding"].values))
index.add(normalized_embeddings)

# Function to search for similar articles
def search_trends(query, k=2):
    query_embedding = get_embedding(query).reshape(1, -1)
    normalized_query = normalize(query_embedding)
    
    distances, indices = index.search(normalized_query, k)
    
    results = []
    for i in indices[0]:
        results.append({"title": df.iloc[i]["title"], "text": df.iloc[i]["text"][:300] + "..."})  # Preview
    
    return results

# Example search
query = "Drug smuggling in prisons"
top_matches = search_trends(query)

# Output results
print("\nTop Related Crime Trends:")
for i, match in enumerate(top_matches, 1):
    print(f"{i}. {match['title']}")
    print(f"   Preview: {match['text']}\n")
