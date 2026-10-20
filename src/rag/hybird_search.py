from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain import Document
from typing import List



def hybrid_retriever( query , retrievers , weights , k=3 , rrf_k=60 ):
    doc_scores = {}
    for retriever ,weight  in zip( retrievers, weights ):
        results = retriever.invoke(query)
        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1.0 / (rank + rrf_k))

            if key in doc_scores:
                doc_scores[key] = (doc_scores[key][0] + rrf_score,doc)
            else:
                doc_scores[key] = (rrf_score,doc)


    sorted_docs = sorted(doc_scores.values(), key=lambda x:x[0] ,reverse=True)

    return [doc for _,doc in sorted_docs[:k]]

documents = [
    # Products
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports gigabit speeds and advanced QoS features.",
        metadata={"type": "product"}
    ),
    Document(
        page_content="Product SKU-8810A is a dual-band Wi-Fi 6 router designed for homes with multiple connected devices.",
        metadata={"type": "product"}
    ),
    Document(
        page_content="Product SKU-5520B is a compact network switch with 8 Gigabit Ethernet ports for small offices.",
        metadata={"type": "product"}
    ),

    # Troubleshooting
    Document(
        page_content="For network connectivity issues, first check the Ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"}
    ),
    Document(
        page_content="If your router frequently disconnects, restart it and check for firmware updates before resetting it.",
        metadata={"type": "troubleshooting"}
    ),
    Document(
        page_content="Slow Wi-Fi speeds may be caused by interference, outdated firmware, or too many connected devices.",
        metadata={"type": "troubleshooting"}
    ),

    # Errors
    Document(
        page_content="Error code E_CONN_REFUSED indicates that the server rejected the connection. Check firewall settings.",
        metadata={"type": "error"}
    ),
    Document(
        page_content="Error code E_DNS_LOOKUP means the hostname could not be resolved. Verify DNS server settings.",
        metadata={"type": "error"}
    ),
    Document(
        page_content="Error code E_AUTH_401 indicates that authentication failed because credentials are missing or invalid.",
        metadata={"type": "error"}
    ),

    # Authentication
    Document(
        page_content="The authentication process requires valid credentials. Use OAuth2 for secure API access.",
        metadata={"type": "auth"}
    ),
    Document(
        page_content="API keys must be stored in environment variables and must never be committed to a public repository.",
        metadata={"type": "auth"}
    ),
    Document(
        page_content="When an access token expires, obtain a new token using the configured refresh-token workflow.",
        metadata={"type": "auth"}
    ),

    # Configuration
    Document(
        page_content="Router configuration guide: Access the admin panel at 192.168.1.1 to modify network settings.",
        metadata={"type": "configuration"}
    ),
    Document(
        page_content="Configure WPA3 security and use a strong Wi-Fi password to protect your wireless network.",
        metadata={"type": "configuration"}
    ),
    Document(
        page_content="DHCP automatically assigns IP addresses to devices on a network. Check the DHCP pool if devices cannot connect.",
        metadata={"type": "configuration"}
    ),

    # Billing and support
    Document(
        page_content="The standard router warranty lasts two years from the date of purchase and covers manufacturing defects.",
        metadata={"type": "warranty"}
    ),
    Document(
        page_content="Customers can request a replacement for a defective router by providing the purchase receipt and serial number.",
        metadata={"type": "support"}
    ),
    Document(
        page_content="Firmware updates include security patches, bug fixes, and performance improvements for supported devices.",
        metadata={"type": "firmware"}
    ),

    # Networking concepts
    Document(
        page_content="DNS translates human-readable domain names into IP addresses that computers use to locate servers.",
        metadata={"type": "networking"}
    ),
    Document(
        page_content="A VPN encrypts network traffic between a device and a VPN endpoint, helping protect data on untrusted networks.",
        metadata={"type": "networking"}
    ),
    Document(
        page_content="Quality of Service (QoS) prioritizes selected network traffic, such as video calls, over less time-sensitive traffic.",
        metadata={"type": "networking"}
    ),
]


class HybridRetriever:
    
    def __init__(self, documents:list[Document] , bm25_weight:float=0.5 , k:int=4):
            self.k=k
            self.bm25_weight=bm25_weight
            self.vector_weight=1-bm25_weight

            self.embeddings= GoogleGenerativeAIEmbeddings("gemini-embedding-001")

            # Create vector store and retriever
            self.vectorstore= Chroma.from_documents(
                 documents,self.embeddings,
                 collection_name="hybrid_search"
            )

            # Create vector Retriever

            self.vector_retriever = self.vectorstore.as_retriever(
                 search_kwargs={"k": self.k}
            )

            # Create BM25 Retriever
            self.bm25_retriever = BM25Retriever.from_documents(documents,k=self.k)

def search(self,query:str)->list[Document]:
     return hybrid_retriever(
          query,
          retrievers=[self.bm25_retriever,
                      self.vector_retriever],
          weights=[self.bm25_weight, self.vector_weight],
          k=self.k
     )

#add new documents to both retrievers
def add_documents(self , documents:list[Document]):
     self.vectorstore.add_documents(documents)

     all_docs = self.vectorstore.get()
     self.bm25_retriever=BM25Retriever.from_documents(
          [Document(page_content=doc) for doc in all_docs["documents"]],
          k=self.k
     )

retriever = HybridRetriever(documents,bm25_weight=0.5 ,k=4)
results = retriever.search("sku-7742X spefications")

for doc in results:
     print(doc.page_content[:100])




