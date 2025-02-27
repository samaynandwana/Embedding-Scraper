from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# Load SBERT model (optimized for similarity tasks)
sbert_model = SentenceTransformer("all-mpnet-base-v2")

# List of articles
articles = [
    {
        "id": 1,
        "title": "Fort Wayne Man Sentenced to 84 Months",
        "text": """FORT WAYNE – Yesterday, Hamed A. Martin, 42 years old, of Fort Wayne, Indiana was sentenced by United States District Court Chief Judge Holly Brady after pleading guilty to distributing methamphetamine, announced Acting United States Attorney Tina L. Nommay.
        Martin was sentenced to 84 months in prison followed by 4 years of supervised release.
        According to documents in the case, in July 2022, Martin distributed methamphetamine on several occasions. A search warrant executed at his residence in August 2022, resulted in the recovery of a firearm along with evidence of drug distribution.
        This case was investigated by the Federal Bureau of Investigation’s Fort Wayne Safe Streets Gang Task Force, which includes the FBI, the Indiana State Police, the Allen County Sheriff’s Department, and the Fort Wayne Police Department."""
    },
    {
        "id": 2,
        "title": "Former Prison Guard Sentenced for Smuggling Drugs",
        "text": """BOSTON – A Virginia man was sentenced yesterday for conspiring to distribute controlled substances and launder drug proceeds with co-conspirators in Massachusetts and Virginia.
        Kenneth J. Owen, 24, of Charlotte Court House, Va., was sentenced to 21 months in prison, followed by three years of supervised release. Owen pleaded guilty to conspiracy to distribute MDMA and buprenorphine and two counts of money laundering conspiracy.
        In December 2019 and January 2020, Owen conspired with others to smuggle drugs into Buckingham Correctional Center in Virginia. He received $1,600 in bribes to deliver the drugs. The smuggled drugs were sold to other inmates, and payments were tracked via Cash App."""
    },
    {
        "id": 3,
        "title": "Drug Trafficker with Mexican Cartel Sentenced",
        "text": """LAREDO, Texas – A 37-year-old man has been sentenced for conspiring to distribute a large quantity of marijuana, announced U.S. Attorney Nicholas J. Ganjei.
        Gavino Cadena pleaded guilty Nov. 10, 2022. U.S. District Judge Diana Saldana sentenced him to 194 months in federal prison, followed by five years of supervised release.
        The court considered Cadena’s extensive criminal record, including his involvement with Cartel del Noreste (CDN) and the Tango Blast gang. Authorities seized more than 17 tons of marijuana valued at approximately $16.4 million in this case."""
    }
]

# Extract article texts and titles
article_texts = [article["text"] for article in articles]
article_titles = [article["title"] for article in articles]

# Generate SBERT embeddings
embeddings = sbert_model.encode(article_texts, convert_to_numpy=True)

# Compute similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Display similarity matrix as a dataframe
df_similarity = pd.DataFrame(similarity_matrix, index=article_titles, columns=article_titles)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print("\nArticle Similarity Matrix:")
print(df_similarity)

# Perform clustering (K-Means)
num_clusters = 2  # Adjust based on dataset size
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)

# Display clustered articles
df_clusters = pd.DataFrame({"Title": article_titles, "Cluster": labels})
print("\nClustered Articles:")
print(df_clusters)
